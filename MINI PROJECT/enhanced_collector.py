# enhanced_collector.py
import time
import csv
import threading
from datetime import datetime
from collections import deque, defaultdict
from pynput import keyboard
import pygetwindow as gw   # pip install pygetwindow
import os

OUTPUT_CSV = "live_feature_log.csv"
WINDOW_SECONDS = 60  # snapshot every 60s

backspace_names = {"backspace","key.backspace","Key.backspace","BackSpace","Backspace","KEY_BACKSPACE","Back_Space"}

class Collector:
    def __init__(self, window_sec=WINDOW_SECONDS):
        self.lock = threading.Lock()
        self.events = []  # (t, key, type)
        self.press_times = []   # (t, key)
        self.release_times = [] # (t, key)
        self.window_titles = [] # (t, title)
        self.window_sec = window_sec
        self.running = False
        self.listener = None
        self._last_active_window = None
        self._monitor_thread = None

        # create output csv header if not exists
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp","n_events","n_chars","backspace_count","app_switches",
                    "duration_sec","wpm","avg_hold_sec","std_hold_sec","avg_flight_sec","std_flight_sec","max_idle_sec","score"
                ])

    def _now(self):
        return time.time()

    def _key_to_str(self, key):
        try:
            return key.char if hasattr(key, "char") and key.char is not None else str(key)
        except Exception:
            return str(key)

    def on_press(self, key):
        t = self._now()
        k = self._key_to_str(key)
        with self.lock:
            self.press_times.append((t, k))
            self.events.append((t, k, "press"))

    def on_release(self, key):
        t = self._now()
        k = self._key_to_str(key)
        with self.lock:
            self.release_times.append((t, k))
            self.events.append((t, k, "release"))

    def _monitor_active_window(self):
        # runs in a separate thread to poll active window title
        while self.running:
            try:
                w = gw.getActiveWindow()
                title = w.title if w else "Unknown"
            except Exception:
                title = "Unknown"
            t = self._now()
            with self.lock:
                if not self._last_active_window:
                    self._last_active_window = title
                    self.window_titles.append((t, title))
                else:
                    if title != self._last_active_window:
                        self._last_active_window = title
                        self.window_titles.append((t, title))
            time.sleep(0.8)  # poll ~1 Hz

    def start(self):
        if self.running:
            return
        self.running = True
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._monitor_thread = threading.Thread(target=self._monitor_active_window, daemon=True)
        self._monitor_thread.start()
        self._snapshot_loop_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._snapshot_loop_thread.start()
        print("Collector started.")

    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()
            self.listener = None
        print("Collector stopped.")

    def pop_since(self, cutoff):
        with self.lock:
            ev = [e for e in self.events if e[0] >= cutoff]
            self.events = [e for e in self.events if e[0] >= cutoff]
            press = [p for p in self.press_times if p[0] >= cutoff]
            self.press_times = [p for p in self.press_times if p[0] >= cutoff]
            release = [r for r in self.release_times if r[0] >= cutoff]
            self.release_times = [r for r in self.release_times if r[0] >= cutoff]
            window_titles = [w for w in self.window_titles if w[0] >= cutoff]
            self.window_titles = [w for w in self.window_titles if w[0] >= cutoff]
            return ev, press, release, window_titles

    def _compute_features(self, events, press, release, windows):
        if not events:
            return None
        t0 = events[0][0]
        tN = events[-1][0]
        duration = max(tN - t0, 1e-6)

        keys_pressed = [e[1] for e in events if e[2]=="press"]
        n_events = len(events)
        n_chars = sum(1 for k in keys_pressed if (len(k)==1 and k.isprintable()) or k.lower() in ("key.space","space"))
        backspaces = sum(1 for e in events if e[1] in backspace_names and e[2]=="press")
        # app switches
        app_switches = len(windows)

        wpm = (n_chars/5.0) / (duration/60.0) if duration>0 else 0.0

        # dwell
        from collections import defaultdict, deque
        press_q = defaultdict(deque)
        holds = []
        for t,k in press:
            press_q[k].append(t)
        for t_rel,k in release:
            if press_q[k]:
                t_press = press_q[k].popleft()
                d = t_rel - t_press
                if 0<d<2.0:
                    holds.append(d)
        avg_hold = float(sum(holds)/len(holds)) if holds else 0.0
        std_hold = float((sum((x-avg_hold)**2 for x in holds)/len(holds))**0.5) if holds else 0.0

        press_ts = sorted([t for t,_ in press])
        flights = []
        for i in range(1,len(press_ts)):
            dt = press_ts[i]-press_ts[i-1]
            if 0<dt<10.0:
                flights.append(dt)
        avg_flight = float(sum(flights)/len(flights)) if flights else 0.0
        std_flight = float((sum((x-avg_flight)**2 for x in flights)/len(flights))**0.5) if flights else 0.0
        max_idle = float(max(flights)) if flights else 0.0

        return {
            "n_events": n_events,
            "n_chars": n_chars,
            "backspace_count": backspaces,
            "app_switches": app_switches,
            "duration_sec": duration,
            "wpm": wpm,
            "avg_hold_sec": avg_hold,
            "std_hold_sec": std_hold,
            "avg_flight_sec": avg_flight,
            "std_flight_sec": std_flight,
            "max_idle_sec": max_idle
        }

    def _snapshot_loop(self):
        # create first cutoff
        self._last_snapshot = time.time()
        while self.running:
            time.sleep(self.window_sec)
            now = time.time()
            cutoff = now - self.window_sec
            events, press, release, windows = self.pop_since(cutoff)
            feats = self._compute_features(events, press, release, windows)
            if feats:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # append placeholder score (dashboard/model will compute)
                row = [ts,
                       feats["n_events"],feats["n_chars"],feats["backspace_count"],feats["app_switches"],
                       feats["duration_sec"],feats["wpm"],feats["avg_hold_sec"],feats["std_hold_sec"],
                       feats["avg_flight_sec"],feats["std_flight_sec"],feats["max_idle_sec"],""]
                with open(OUTPUT_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            self._last_snapshot = now

if __name__ == "__main__":
    c = Collector()
    c.start()
    print("Collector running; press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        c.stop()
        print("Stopped.")
