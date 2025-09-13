
import os
import time
import math
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict

import numpy as np
import pandas as pd
from pynput import keyboard
import joblib

# Tkinter + matplotlib (no seaborn, single-plot rule)
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ============ CONFIG ============
MODEL_PATH = r"./fatigue_demo_model.pkl"   # update if needed
OUTPUT_LOG = "fatigue_scores_log.csv"
WINDOW_SECONDS = 60             # compute one score every 60s
BASE_BREAK_MIN = 25             # default break interval (minutes)
HIGH_FATIGUE_BREAK_MIN = 10     # if score >= 0.7, suggest sooner break
TREND_WINDOW_DAYS = 7
# ================================

FEATURE_COLUMNS = [
    "n_events","n_chars","backspace_count","duration_sec","wpm",
    "avg_hold_sec","std_hold_sec","avg_flight_sec","std_flight_sec","max_idle_sec"
]

# ------------ Keystroke Collector ------------
class KeystrokeCollector:
    def __init__(self):
        self.lock = threading.Lock()
        self.press_times = []     # (t, key_str)
        self.release_times = []   # (t, key_str)
        self.events = []          # (t, key_str, event_type)
        self.running = False
        self.listener = None

    def _now(self):
        return time.time()

    def on_press(self, key):
        t = self._now()
        key_str = self._key_to_str(key)
        with self.lock:
            self.press_times.append((t, key_str))
            self.events.append((t, key_str, "press"))

    def on_release(self, key):
        t = self._now()
        key_str = self._key_to_str(key)
        with self.lock:
            self.release_times.append((t, key_str))
            self.events.append((t, key_str, "release"))

    def _key_to_str(self, key):
        try:
            return key.char if hasattr(key, "char") and key.char is not None else str(key)
        except Exception:
            return str(key)

    def start(self):
        if self.running: return
        self.running = True
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def stop(self):
        if not self.running: return
        self.running = False
        if self.listener:
            self.listener.stop()
            self.listener = None

    def pop_events_since(self, ts_cutoff):
        with self.lock:
            new_events = [e for e in self.events if e[0] >= ts_cutoff]
            self.events = new_events
            self.press_times = [p for p in self.press_times if p[0] >= ts_cutoff]
            self.release_times = [r for r in self.release_times if r[0] >= ts_cutoff]
            return list(new_events), list(self.press_times), list(self.release_times)

# ------------ Feature Computation ------------
def compute_features(events, press_times, release_times):
    if not events:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    t0 = events[0][0]
    tN = events[-1][0]
    duration = max(tN - t0, 1e-6)

    keys_pressed = [e[1] for e in events if e[2] == "press"]
    n_events = len(events)

    def is_char_like(k):
        try:
            return (len(k) == 1 and k.isprintable()) or k.lower() in ("key.space","space")
        except Exception:
            return False
    n_chars = sum(1 for k in keys_pressed if is_char_like(k))

    backspace_names = {"backspace","key.backspace","Key.backspace","BackSpace","Backspace","KEY_BACKSPACE","Back_Space"}
    backspaces = sum(1 for e in events if e[1] in backspace_names and e[2]=="press")

    wpm = (n_chars / 5.0) / (duration / 60.0)

    # Dwell times: match press-release per key FIFO
    holds = []
    press_q = defaultdict(deque)
    for t, k in press_times:
        press_q[k].append(t)
    for t_rel, k in release_times:
        if press_q[k]:
            t_press = press_q[k].popleft()
            dwell = t_rel - t_press
            if 0 < dwell < 2.0:
                holds.append(dwell)
    avg_hold = float(np.mean(holds)) if holds else 0.0
    std_hold  = float(np.std(holds)) if holds else 0.0

    press_ts = sorted([t for t, _ in press_times])
    flights = []
    for i in range(1, len(press_ts)):
        dt = press_ts[i] - press_ts[i-1]
        if 0 < dt < 10.0:
            flights.append(dt)
    avg_flight = float(np.mean(flights)) if flights else 0.0
    std_flight = float(np.std(flights)) if flights else 0.0
    max_idle = float(np.max(flights)) if flights else 0.0

    return {
        "n_events": float(n_events),
        "n_chars": float(n_chars),
        "backspace_count": float(backspaces),
        "duration_sec": float(duration),
        "wpm": float(wpm),
        "avg_hold_sec": float(avg_hold),
        "std_hold_sec": float(std_hold),
        "avg_flight_sec": float(avg_flight),
        "std_flight_sec": float(std_flight),
        "max_idle_sec": float(max_idle),
    }

# ------------ Real-time Engine ------------
class RealTimeFatigueEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)
        self.collector = KeystrokeCollector()
        self.window_sec = WINDOW_SECONDS
        self.last_compute = time.time()
        self.log_path = OUTPUT_LOG
        self.history = deque(maxlen=200)  # ~200 minutes

    def start(self):
        self.collector.start()
        self.last_compute = time.time()

    def stop(self):
        self.collector.stop()

    def step(self):
        now = time.time()
        if now - self.last_compute >= self.window_sec:
            cutoff = now - self.window_sec
            events, press_times, release_times = self.collector.pop_events_since(cutoff)
            feats = compute_features(events, press_times, release_times)
            X = np.array([[feats[c] for c in FEATURE_COLUMNS]], dtype=float)
            try:
                prob1 = float(self.model.predict_proba(X)[0,1])
            except Exception:
                prob1 = float(self.model.predict(X)[0])
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {"timestamp": ts, **feats, "score": prob1}
            self.history.append(row)
            self.append_log(row)
            self.last_compute = now
            return row
        return None

    def append_log(self, row):
        import csv, os
        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp"]+FEATURE_COLUMNS+["score"])
            if write_header:
                w.writeheader()
            w.writerow(row)

    def load_log_df(self):
        if not os.path.exists(self.log_path):
            return pd.DataFrame(columns=["timestamp","score"])
        df = pd.read_csv(self.log_path)
        # ensure timestamp dtype
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        return df

# ------------ Prescriptions ------------
def make_prescription(score, recent_scores):
    tips = []
    # Always give a core set
    tips.append("Follow the 20–20–20 rule for eyes (every 20 min, look 20 ft away for 20 sec).")
    tips.append("Sip water; mild dehydration increases perceived fatigue.")
    tips.append("2 min shoulder/neck stretch + 1 min wrist flexor stretch.")
    # Score-based
    if score >= 0.7:
        tips.append("Take a 5–10 minute break: stand up, walk, or do light movement.")
        tips.append("Try 1 minute of slow breathing (inhale 4s, exhale 6s).")
        tips.append("If high for multiple windows, consider a 20–30 min rest or power nap.")
    elif score >= 0.5:
        tips.append("Microbreak: 2–3 minutes standing + gentle stretches.")
    # Trend-based
    if len(recent_scores) >= 3 and np.mean(recent_scores[-3:]) >= 0.7:
        tips.append("Sustained high fatigue detected. Reduce cognitive load or reschedule heavy tasks.")
    return "\n• " + "\n• ".join(tips)

# ------------ Tkinter Dashboard with Trend & Break Timer ------------
class DashboardApp:
    def __init__(self, engine: RealTimeFatigueEngine):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Real-time Fatigue Detector")
        self.root.geometry("820x680")

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.status_var = tk.StringVar(value="Status: Idle")
        self.score_var  = tk.StringVar(value="Score: 0.00")
        self.advice_var = tk.StringVar(value="—")
        self.break_var  = tk.StringVar(value="Next break in: --:--")
        self.next_break_time = None

        title = ttk.Label(self.root, text="Real-time Fatigue Detector", font=("Segoe UI", 16, "bold"))
        title.pack(pady=8)

        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=5, fill="x")

        self.score_label = ttk.Label(top_frame, textvariable=self.score_var, font=("Segoe UI", 14))
        self.score_label.pack(side="left", padx=10)

        self.progress = ttk.Progressbar(top_frame, orient="horizontal", length=300, mode="determinate", maximum=100)
        self.progress.pack(side="left", padx=10)

        self.status_label = ttk.Label(top_frame, textvariable=self.status_var)
        self.status_label.pack(side="left", padx=10)

        # Break reminder row
        br_frame = ttk.Frame(self.root)
        br_frame.pack(pady=6, fill="x")
        ttk.Label(br_frame, textvariable=self.break_var, font=("Segoe UI", 12)).pack(side="left", padx=10)
        ttk.Button(br_frame, text="I took a break", command=self.on_break_taken).pack(side="left", padx=6)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=6)
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.on_start)
        self.start_btn.grid(row=0, column=0, padx=8)
        self.stop_btn  = ttk.Button(btn_frame, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=8)

        # Table
        self.tree = ttk.Treeview(self.root, columns=("time","score"), show="headings", height=8)
        self.tree.heading("time", text="Timestamp")
        self.tree.heading("score", text="Fatigue Score")
        self.tree.column("time", width=220)
        self.tree.column("score", width=120, anchor="center")
        self.tree.pack(pady=6, fill="x")

        # Prescription panel
        pres_frame = ttk.LabelFrame(self.root, text="Prescription (next actions)")
        pres_frame.pack(pady=6, fill="both", expand=True)
        self.pres_text = tk.Text(pres_frame, height=6, wrap="word")
        self.pres_text.pack(fill="both", expand=True, padx=8, pady=6)
        self.pres_text.insert("1.0", "Start the detector to receive personalized guidance.")

        # Matplotlib Figure (7-day trend)
        fig_frame = ttk.LabelFrame(self.root, text="7‑Day Trend (scores per minute)")
        fig_frame.pack(pady=6, fill="both", expand=True)
        self.fig = plt.figure(figsize=(6, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # periodic loop
        self.root.after(500, self.loop)

    def on_start(self):
        self.engine.start()
        self.status_var.set("Status: Collecting...")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.schedule_next_break(initial=True)

    def on_stop(self):
        self.engine.stop()
        self.status_var.set("Status: Stopped")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def on_break_taken(self):
        # Reset break timer and note the time
        self.schedule_next_break(force_base=True)
        messagebox.showinfo("Break", "Great! Break recorded. Timer reset.")

    def schedule_next_break(self, initial=False, force_base=False):
        now = datetime.now()
        # If force_base or at start, schedule base interval
        if force_base or initial:
            self.next_break_time = now + timedelta(minutes=BASE_BREAK_MIN)
            return
        # If last score high, schedule sooner break
        last_score = self.get_last_score()
        if last_score is not None and last_score >= 0.7:
            self.next_break_time = now + timedelta(minutes=HIGH_FATIGUE_BREAK_MIN)
        else:
            self.next_break_time = now + timedelta(minutes=BASE_BREAK_MIN)

    def get_last_score(self):
        if not len(self.engine.history):
            return None
        return self.engine.history[-1]["score"]

    def update_break_countdown(self):
        if not self.next_break_time:
            self.break_var.set("Next break in: --:--")
            return
        now = datetime.now()
        delta = self.next_break_time - now
        if delta.total_seconds() <= 0:
            self.break_var.set("Next break in: 00:00")
            # Nudge user
            self.advice_var.set("⏰ Break time! Stand, stretch, hydrate.")
            try:
                self.root.bell()
            except Exception:
                pass
            # After alert, schedule next interval based on latest score
            self.schedule_next_break()
        else:
            mins = int(delta.total_seconds() // 60)
            secs = int(delta.total_seconds() % 60)
            self.break_var.set(f"Next break in: {mins:02d}:{secs:02d}")

    def update_prescription(self):
        last_score = self.get_last_score() or 0.0
        recent = [r["score"] for r in list(self.engine.history)[-10:]]
        text = make_prescription(last_score, recent)
        self.pres_text.delete("1.0", "end")
        self.pres_text.insert("1.0", text)

    def update_table(self, row):
        self.tree.insert("", "end", values=(row["timestamp"], f"{row['score']:.2f}"))
        if len(self.tree.get_children()) > 30:
            first = self.tree.get_children()[0]
            self.tree.delete(first)

    def update_chart(self):
        df = self.engine.load_log_df()
        if df.empty or "score" not in df.columns:
            self.ax.clear()
            self.ax.set_title("No data yet")
            self.canvas.draw()
            return
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=TREND_WINDOW_DAYS)
        df = df[df["timestamp"] >= cutoff]
        self.ax.clear()
        # single-plot line of scores over time
        self.ax.plot(df["timestamp"], df["score"], marker='o', linewidth=1)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Fatigue score")
        self.ax.set_xlabel("Time")
        self.ax.set_title("Last 7 days")
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def loop(self):
        row = self.engine.step()
        if row is not None:
            score = row["score"]
            self.score_var.set(f"Score: {score:.2f}")
            self.progress["value"] = int(round(score * 100))

            if score >= 0.7:
                self.advice_var.set("⚠️ High fatigue predicted. Consider a short break.")
            elif score >= 0.5:
                self.advice_var.set("🙂 Moderate fatigue. Stay mindful, hydrate.")
            else:
                self.advice_var.set("✅ Low fatigue. Keep it up!")

            self.update_table(row)
            self.update_prescription()
            # reschedule next break based on latest score as it arrives
            self.schedule_next_break()

        self.update_break_countdown()
        self.update_chart()
        self.root.after(1000, self.loop)

    def run(self):
        self.root.mainloop()

def main():
    engine = RealTimeFatigueEngine(MODEL_PATH)
    app = DashboardApp(engine)
    app.run()

if __name__ == "__main__":
    main()
