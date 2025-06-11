import os
import pickle
import time

import cvzone
import numpy as np
import cv2
import face_recognition

from encodingGenerator import encodeListKnownWithIds

class FaceAttendanceSystem:
    def __init__(
        self,
        video_index=1,
        background_path='Resources/background.png',
        modes_folder='Resources/Modes',
        encode_file='EncodeFile.p',
        window=2,
        absent_threshold=30,
        persist_file='attendance_data.p',
        verbose=False
    ):
        """
        Initialize the face attendance system.

        Parameters:
        - video_index: int, index for cv2.VideoCapture.
        - background_path: path to background image.
        - modes_folder: folder path containing mode images.
        - encode_file: path to pickle file containing (encodeListKnown, faceIDs).
        - window: evaluation window in seconds (2).
        - absent_threshold: seconds of continuous absence to mark and accumulate (30).
        - persist_file: filename to persist attendance data (worked + total absent) on exit.
        - verbose: if True, prints debug logs about timing and attendance events.
        """
        self.verbose = verbose

        # Webcam setup
        self.cap = cv2.VideoCapture(video_index)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # Paths and files
        self.background_path = background_path
        self.modes_folder = modes_folder
        self.encode_file = encode_file
        self.persist_file = persist_file

        # Attendance window settings
        self.window = window
        self.absent_threshold = absent_threshold

        # Load and validate resources
        self._load_background()
        self._load_mode_images()
        self._load_encodings()

        # Initialize per-user attendance state
        # worked_time and absent_time in seconds (float)
        self.worked_time = {uid: 0.0 for uid in self.faceIDs}
        self.absent_time = {uid: 0.0 for uid in self.faceIDs}
        self.total_absent_time = {uid: 0.0 for uid in self.faceIDs}
        self.detected_in_window = {uid: False for uid in self.faceIDs}

        # Window timing: will be aligned when run() starts processing frames
        self.window_start = None

        # Load persisted data if exists
        self._load_persisted_data()

        print(f"[Init] Users: {self.faceIDs}")
        print(f"[Init] Window: {self.window}s, Absent threshold: {self.absent_threshold}s")

    def _load_background(self):
        if not os.path.isfile(self.background_path):
            raise FileNotFoundError(f"Background image not found at {self.background_path}")
        # We reload it each frame in run()

    def _load_mode_images(self):
        if not os.path.isdir(self.modes_folder):
            raise FileNotFoundError(f"Modes folder not found at {self.modes_folder}")
        self.imgModeList = []
        for fname in sorted(os.listdir(self.modes_folder)):
            path = os.path.join(self.modes_folder, fname)
            img = cv2.imread(path)
            if img is not None:
                self.imgModeList.append(img)
        if len(self.imgModeList) == 0:
            print(f"[Warning] No mode images loaded from {self.modes_folder}")

    def _load_encodings(self):
        if not os.path.isfile(self.encode_file):
            raise FileNotFoundError(f"Encode file not found at {self.encode_file}")
        with open(self.encode_file, 'rb') as f:
            data = pickle.load(f)
        if not (isinstance(data, (list, tuple)) and len(data) == 2):
            raise ValueError("EncodeFile.p must contain (encodeListKnown, faceIDs)")
        self.encodeListKnown, self.faceIDs = data
        if not isinstance(self.encodeListKnown, list) or not isinstance(self.faceIDs, list):
            raise ValueError("encodeListKnown must be list of encodings, faceIDs a list of IDs")

    def _load_persisted_data(self):
        """
        Load persisted worked_time and total_absent_time if exists.
        Supports two formats:
         - Older: pickle contains dict of worked_time only.
         - Newer: pickle contains dict with keys 'worked_time' and 'total_absent_time'.
        """
        if os.path.isfile(self.persist_file):
            try:
                with open(self.persist_file, 'rb') as f:
                    data = pickle.load(f)
                # Check format
                if isinstance(data, dict):
                    # New format?
                    if 'worked_time' in data and 'total_absent_time' in data:
                        prev_wt = data['worked_time']
                        prev_atot = data['total_absent_time']
                        for uid, wt in prev_wt.items():
                            if uid in self.worked_time:
                                self.worked_time[uid] = float(wt)
                        for uid, atot in prev_atot.items():
                            if uid in self.total_absent_time:
                                self.total_absent_time[uid] = float(atot)
                        print(f"[Info] Loaded persisted worked_time & total_absent_time from {self.persist_file}")
                    else:
                        # Assume older format: worked_time dict only
                        for uid, wt in data.items():
                            if uid in self.worked_time:
                                self.worked_time[uid] = float(wt)
                        print(f"[Info] Loaded persisted worked_time from {self.persist_file} (old format)")
                else:
                    print(f"[Warning] Persisted data format not recognized. Skipping load.")
            except Exception as e:
                print(f"[Warning] Could not load persisted data: {e}")

    def mark_detected(self, uid):
        """Mark that user `uid` was detected in the current window."""
        if uid in self.detected_in_window:
            self.detected_in_window[uid] = True

    def time_to_evaluate(self):
        """
        Return True if current time >= window_start + window length,
        indicating the window elapsed.
        """
        return time.monotonic() >= self.window_start + self.window

    def evaluate_window(self):
        """
        Evaluate attendance at end of window using actual elapsed time.
        Returns:
          events: {
            'forgiven_absences': [(uid, forgiven_seconds), ...],
            'worked_added':        [(uid, added_seconds), ...],
            'marked_absence':      [(uid, absent_threshold), ...]
          }
        """
        now = time.monotonic()
        actual_interval = now - self.window_start
        if self.verbose:
            print(f"[Eval] Window elapsed {actual_interval:.2f}s (threshold {self.window}s)")

        events = {
            'forgiven_absences': [],
            'worked_added': [],
            'marked_absence': []
        }

        for uid in self.faceIDs:
            if self.detected_in_window.get(uid, False):
                # Seen in window
                if self.absent_time[uid] > 0:
                    forgiven = self.absent_time[uid]
                    self.worked_time[uid] += forgiven
                    events['forgiven_absences'].append((uid, forgiven))
                    if self.verbose:
                        print(f"  [Forgive] {uid}: +{forgiven:.2f}s (absent forgiven)")
                    self.absent_time[uid] = 0.0
                # Add actual interval to worked_time
                self.worked_time[uid] += actual_interval
                events['worked_added'].append((uid, actual_interval))
                if self.verbose:
                    print(f"  [Work+] {uid}: +{actual_interval:.2f}s worked (total {self.worked_time[uid]:.2f}s)")
            else:
                # Not seen
                before = self.absent_time[uid]
                self.absent_time[uid] += actual_interval
                after = self.absent_time[uid]
                if self.verbose:
                    print(f"  [Absent] {uid}: absent_time was {before:.2f}s, now {after:.2f}s")
                if after >= self.absent_threshold:
                    # Mark a full absence event
                    events['marked_absence'].append((uid, self.absent_threshold))
                    # Accumulate total absent time
                    self.total_absent_time[uid] += self.absent_threshold
                    if self.verbose:
                        print(f"    [Mark] {uid}: absent_time {after:.2f}s >= threshold {self.absent_threshold}s → marked absence, total_absent_time now {self.total_absent_time[uid]:.2f}s")
                    # Reset absent_time
                    self.absent_time[uid] = 0.0
                # else: keep accumulating for next window

        # Advance window_start to now
        self.window_start = now
        # Reset detection flags
        for uid in self.faceIDs:
            self.detected_in_window[uid] = False

        if self.verbose:
            print(f"[Eval] Next window_start set to {self.window_start:.2f}\n")
        return events

    def _format_time(self, total_seconds):
        """
        Format seconds to H:MM:SS or M:SS depending on length.
        """
        total_seconds = int(total_seconds)
        hrs = total_seconds // 3600
        mins = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        if hrs > 0:
            return f"{hrs}:{mins:02d}:{secs:02d}"
        else:
            return f"{mins:02d}:{secs:02d}"

    def run(self):
        """
        Main loop: capture frames, detect faces, update attendance, overlay UI, display.
        Press 'q' to exit. On exit, persists attendance data.
        """
        # Warm up camera and align window_start
        success, frame = self.cap.read()
        if not success:
            print("[Error] Camera read failed at start. Exiting.")
            return
        # Align the first window start here
        self.window_start = time.monotonic()
        print(f"[Run] Window aligned at {time.ctime()} (monotonic {self.window_start:.2f})")

        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("[Error] Camera read failed. Exiting loop.")
                    break

                # Resize & convert for faster face recognition
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                # Detect faces & encodings
                faceLocs = face_recognition.face_locations(imgS)
                faceEncodings = face_recognition.face_encodings(imgS, faceLocs)

                # Collect detected IDs and bboxes
                presentIDs = set()
                bboxes_full = []
                for enc, loc in zip(faceEncodings, faceLocs):
                    matches = face_recognition.compare_faces(self.encodeListKnown, enc)
                    distances = face_recognition.face_distance(self.encodeListKnown, enc)
                    if len(distances) == 0:
                        continue
                    idx = np.argmin(distances)
                    if matches[idx]:
                        uid = self.faceIDs[idx]
                        presentIDs.add(uid)
                        self.mark_detected(uid)
                        # Compute full-size bbox:
                        y1, x2, y2, x1 = loc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        bbox = (55 + x1, 162 + y1, x2 - x1, y2 - y1)
                        bboxes_full.append((uid, bbox))

                # Evaluate window if needed
                if self.time_to_evaluate():
                    events = self.evaluate_window()
                    # Optionally log events (especially if verbose=False, you may want to log major ones)
                    if not self.verbose:
                        now_str = time.ctime()
                        for uid, forgiven in events['forgiven_absences']:
                            print(f"{now_str}: {uid} returned → forgiven {forgiven:.2f}s absence.")
                        for uid, added in events['worked_added']:
                            print(f"{now_str}: {uid} detected → +{added:.2f}s worked.")
                        for uid, thr in events['marked_absence']:
                            print(f"{now_str}: {uid} absent {thr}s → marked absence, total_absent_time now {self.total_absent_time[uid]:.2f}s.")

                # Overlay UI
                imgBackground = cv2.imread(self.background_path)
                if imgBackground is None:
                    print(f"[Error] Background not found at {self.background_path}. Exiting.")
                    break

                # Paste camera feed region (adjust region as per your background design)
                imgBackground[162:162 + 480, 55:55 + 640] = img

                # Paste a mode image if available (e.g., index 1)
                if len(self.imgModeList) > 1 and self.imgModeList[1] is not None:
                    imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[1]

                # Draw bounding boxes & names
                for uid, bbox in bboxes_full:
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    x, y, w, h = bbox
                    # Draw name with background rect
                    cvzone.putTextRect(
                        imgBackground, uid,
                        (x, y - 10),
                        scale=0.7, thickness=2,
                        colorR=(255, 255, 255), colorT=(0, 0, 0)
                    )

                # Draw a semi-transparent side panel for attendance info
                panel_width = 350
                h_img, w_img, _ = imgBackground.shape
                overlay = imgBackground.copy()
                panel_color = (30, 30, 30)  # darker gray
                alpha = 0.6
                cv2.rectangle(overlay, (0, 0), (panel_width, h_img), panel_color, -1)
                cv2.addWeighted(overlay, alpha, imgBackground, 1 - alpha, 0, imgBackground)

                # Header on panel
                header_text = "Attendance"
                cvzone.putTextRect(
                    imgBackground, header_text,
                    (10, 25),
                    scale=0.8, thickness=2,
                    colorR=(255, 255, 255), colorT=(0, 0, 0)
                )
                # Underline
                cv2.line(imgBackground, (10, 35), (panel_width - 10, 35), (200, 200, 200), 1)

                # Overlay Worked, Current Absent, Total Absent, and absent progress bar per user
                start_x = 10
                start_y = 50
                line_h = 40
                bar_height = 8
                bar_width_max = panel_width - 20  # leave some margin
                for idx, uid in enumerate(self.faceIDs):
                    y_text = start_y + idx * line_h
                    wt = self.worked_time.get(uid, 0.0)
                    at = self.absent_time.get(uid, 0.0)
                    atot = self.total_absent_time.get(uid, 0.0)
                    wt_str = self._format_time(wt)
                    at_str = self._format_time(at)
                    atot_str = self._format_time(atot)
                    text = f"{uid}  W:{wt_str}  Acur:{at_str}  Atot:{atot_str}"
                    cvzone.putTextRect(
                        imgBackground, text,
                        (start_x, y_text),
                        scale=0.6, thickness=1,
                        colorR=(255, 255, 255), colorT=(0, 0, 0)
                    )
                    # Draw absent progress bar under this line
                    # Proportion = absent_time / absent_threshold (clamped to 1.0)
                    prop = min(at / self.absent_threshold, 1.0)
                    bar_x = start_x
                    bar_y = y_text + 20
                    filled_w = int(bar_width_max * prop)
                    # Background of bar (empty)
                    cv2.rectangle(imgBackground, (bar_x, bar_y), (bar_x + bar_width_max, bar_y + bar_height), (100, 100, 100), 1)
                    # Filled portion
                    cv2.rectangle(imgBackground, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_height), (0, 0, 255), -1)

                # Overlay countdown to next evaluation at bottom of panel
                secs_to_next = int((self.window_start + self.window) - time.monotonic())
                if secs_to_next < 0:
                    secs_to_next = 0
                countdown_text = f"Next in: {secs_to_next}s"
                cvzone.putTextRect(
                    imgBackground, countdown_text,
                    (10, h_img - 20),
                    scale=0.6, thickness=1,
                    colorR=(0, 255, 0), colorT=(0, 0, 0)
                )

                # Show the final UI
                cv2.imshow('Face Attendance', imgBackground)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Run] Exit requested. Breaking loop.")
                    break

        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            # Persist attendance data: both worked_time and total_absent_time
            data_to_persist = {
                'worked_time': self.worked_time,
                'total_absent_time': self.total_absent_time
            }
            try:
                with open(self.persist_file, 'wb') as f:
                    pickle.dump(data_to_persist, f)
                print(f"[Save] Saved attendance data to {self.persist_file}")
            except Exception as e:
                print(f"[Warning] Failed to save attendance data: {e}")

    def get_worked_time(self, uid):
        return self.worked_time.get(uid, 0.0)

    def get_absent_time(self, uid):
        return self.absent_time.get(uid, 0.0)

    def get_total_absent_time(self, uid):
        return self.total_absent_time.get(uid, 0.0)


if __name__ == "__main__":
    # Example usage: adjust indices/paths as needed
    fas = FaceAttendanceSystem(
        video_index=1,
        background_path='Resources/background.png',
        modes_folder='Resources/Modes',
        encode_file='EncodeFile.p',
        window=2,
        absent_threshold=30,
        persist_file='attendance_data.p',
        verbose=False  # set True to see detailed debug logs
    )
    fas.run()
