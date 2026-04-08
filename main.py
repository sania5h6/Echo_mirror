"""
EchoMirror - System Controller (main.py)
The unified entry point that ties all modules together:
  - Camera + EmotionDetector (background thread)
  - ConversationEngine + ChatUI (main thread)
  - Explainable AI (injected into both HUD and AI prompt)
  - Agentic AI (crisis detection, interventions)
  - Memory DB (encrypted session storage)
  - Sacred Wisdom (multi-faith healing quotes)

Usage:
  python main.py              → Full system (camera + conversation)
  python main.py --no-camera  → Conversation only (no webcam)
  python main.py --camera     → Camera HUD only (no conversation)
  python main.py --visualize  → Show emotion charts
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import threading
import time
import signal
from datetime import datetime


# ─────────────────────────────────────────
# MODULE IMPORTS
# ─────────────────────────────────────────
def _import_modules():
    """Import all EchoMirror modules with error handling."""
    modules = {}

    try:
        from database import MemoryDB
        modules["db"] = MemoryDB()
        print("[Main] ✓ MemoryDB loaded")
    except Exception as e:
        print(f"[Main] ✗ MemoryDB failed: {e}")
        modules["db"] = None

    try:
        from emotion_detection import EmotionDetector
        modules["EmotionDetector"] = EmotionDetector
        print("[Main] ✓ EmotionDetector loaded")
    except Exception as e:
        print(f"[Main] ✗ EmotionDetector failed: {e}")
        modules["EmotionDetector"] = None

    try:
        from explainable_ai import EmotionExplainer
        modules["xai"] = EmotionExplainer()
        print("[Main] ✓ Explainable AI loaded")
    except Exception as e:
        print(f"[Main] ✗ Explainable AI failed: {e}")
        modules["xai"] = None

    try:
        from agentic_ai import AgenticAI
        modules["agent"] = AgenticAI(db=modules.get("db"))
        print("[Main] ✓ Agentic AI loaded")
    except Exception as e:
        print(f"[Main] ✗ Agentic AI failed: {e}")
        modules["agent"] = None

    return modules


# ─────────────────────────────────────────
# CAMERA THREAD
# ─────────────────────────────────────────
class CameraThread:
    """
    Runs emotion detection in a background thread.
    Provides get_emotion() for other modules to consume.
    """
    def __init__(self, detector, xai=None):
        self.detector = detector
        self.xai = xai
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[CameraThread] Started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[CameraThread] Stopped")

    def _run(self):
        """Main camera loop — runs emotion detection with HUD."""
        import cv2
        from emotion_detection import (
            get_brightness, enhance_low_light, get_nearest_face,
            draw_overlay, LOW_LIGHT_THRESHOLD
        )

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[CameraThread] ERROR: Cannot open camera")
            return

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                self.detector.update_fps()

                brightness = get_brightness(frame)
                self.detector.brightness = brightness
                self.detector.is_low_light = brightness < LOW_LIGHT_THRESHOLD
                proc_frame = (enhance_low_light(frame)
                              if self.detector.is_low_light else frame.copy())

                all_faces = self.detector.detect_faces(proc_frame)
                nearest_face = get_nearest_face(all_faces)

                if nearest_face is not None:
                    now = time.time()
                    should_detect = (now - self.detector.last_detection_time
                                    >= self.detector._detection_interval)
                    face_moved = self.detector.has_face_moved(nearest_face)

                    if should_detect or face_moved:
                        self.detector.last_detection_time = now
                        self.detector.last_face_pos = nearest_face

                        x, y, fw, fh = nearest_face
                        pad_x = int(fw * 0.15)
                        pad_y = int(fh * 0.15)
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(proc_frame.shape[1], x + fw + pad_x)
                        y2 = min(proc_frame.shape[0], y + fh + pad_y)
                        face_crop = proc_frame[y1:y2, x1:x2].copy()
                        threading.Thread(
                            target=self.detector.analyze_emotion,
                            args=(face_crop,), daemon=True
                        ).start()

                        # Feed XAI explainer
                        if self.xai:
                            emotion, scores, _, confidence = self.detector.get_state()
                            if scores:
                                self.xai.explain_detection(emotion, scores, confidence)

                emotion, scores, affirmation, confidence = self.detector.get_state()
                frame = draw_overlay(
                    frame, nearest_face, emotion, scores, affirmation,
                    confidence, self.detector.is_low_light, brightness,
                    self.detector.fps, self.detector.recent_emotions,
                    detector=self.detector
                )

                # XAI overlay on HUD
                if self.xai:
                    xai_lines = self.xai.get_hud_text()
                    h = frame.shape[0]
                    for i, line in enumerate(xai_lines):
                        y_pos = h - 100 - (len(xai_lines) - i) * 22
                        cv2.putText(frame, line, (15, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                    (180, 160, 255), 1)

                cv2.imshow("EchoMirror", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

        except Exception as e:
            print(f"[CameraThread] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[CameraThread] Camera released")

    def get_emotion(self):
        """Get the conversation-level emotion (stable)."""
        return self.detector.get_conversation_emotion()

    def get_display_emotion(self):
        """Get the display-level emotion (fast)."""
        return self.detector.get_current_emotion()


# ─────────────────────────────────────────
# SESSION MANAGER
# ─────────────────────────────────────────
class SessionManager:
    """Manages the full EchoMirror session lifecycle."""

    def __init__(self, modules):
        self.db = modules.get("db")
        self.xai = modules.get("xai")
        self.agent = modules.get("agent")
        self.EmotionDetector = modules.get("EmotionDetector")
        self.camera_thread = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start_time = time.time()

    def start_camera(self):
        """Launch the camera thread."""
        if self.EmotionDetector is None:
            print("[Session] Skipping camera — EmotionDetector not available")
            return False

        detector = self.EmotionDetector(use_db=True)
        self.camera_thread = CameraThread(detector, self.xai)
        self.camera_thread.start()
        return True

    def start_conversation(self):
        """Launch the conversation UI (blocks — runs in main thread)."""
        try:
            import whisper
            import torch
            from conversation import ConversationEngine, EchoMirrorChatUI

            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Session] Loading Whisper on {DEVICE}...")
            whisper_model = whisper.load_model("medium", device=DEVICE)
            print("[Session] Whisper ready")

            engine = ConversationEngine()

            # If camera is running, feed emotion to the engine
            if self.camera_thread:
                # Install a periodic emotion updater
                def _update_emotion():
                    while self.camera_thread._running:
                        try:
                            emo, conf, group = self.camera_thread.get_emotion()
                            engine.emotion = emo
                            # Also feed XAI context into memory_context
                            if self.xai:
                                engine._xai_context = self.xai.get_context_for_prompt()
                        except Exception:
                            pass
                        time.sleep(2)

                threading.Thread(target=_update_emotion, daemon=True).start()
                print("[Session] Camera → Conversation feed active")

            app = EchoMirrorChatUI(engine, whisper_model)
            print("[Session] Launching UI...")
            app.run()  # Blocks until window closed

        except Exception as e:
            print(f"[Session] Conversation error: {e}")
            import traceback
            traceback.print_exc()

    def shutdown(self):
        """Clean shutdown of all modules."""
        duration = time.time() - self._start_time
        print(f"\n[Session] Shutting down (session: {duration:.0f}s)...")

        # Stop camera
        if self.camera_thread:
            self.camera_thread.stop()

        # Save daily summary
        if self.agent:
            self.agent.save_session_summary("neutral", 0.0)

        # Save session to DB
        if self.db:
            try:
                self.db.save_daily_summary()
                print("[Session] Daily summary saved")
            except Exception as e:
                print(f"[Session] Summary save failed: {e}")

        print("[Session] Goodbye. Take care of yourself. 💙")


# ─────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  🪞 EchoMirror — Personal Emotional Reflection System")
    print("  Team: R. Manoj Naik | R. Indu | SK. Sania")
    print("=" * 55)
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode    : ", end="")

    # Parse args
    args = sys.argv[1:]
    camera_only = "--camera" in args
    no_camera = "--no-camera" in args
    visualize = "--visualize" in args

    if visualize:
        print("Visualization")
        print("=" * 55)
        from visualize import show_all_charts
        show_all_charts()
        return

    if camera_only:
        print("Camera Only")
    elif no_camera:
        print("Conversation Only (no webcam)")
    else:
        print("Full System (Camera + Conversation)")
    print("=" * 55)

    # Load modules
    print("\n[Main] Loading modules...")
    modules = _import_modules()
    print()

    # Create session
    session = SessionManager(modules)

    # Handle graceful shutdown
    def _signal_handler(sig, frame):
        session.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        if camera_only:
            # Camera HUD only
            if session.start_camera():
                print("[Main] Camera running. Press 'q' in the window to quit.")
                while session.camera_thread._running:
                    time.sleep(0.5)
            else:
                print("[Main] Camera failed to start.")

        elif no_camera:
            # Conversation only
            session.start_conversation()

        else:
            # Full system — camera + conversation
            camera_ok = session.start_camera()
            if camera_ok:
                print("[Main] Camera started. Launching conversation...\n")
                time.sleep(1)  # Let camera warm up
            else:
                print("[Main] Camera unavailable — running conversation only\n")

            session.start_conversation()  # Blocks until UI closed

    except KeyboardInterrupt:
        pass
    finally:
        session.shutdown()


if __name__ == "__main__":
    main()
