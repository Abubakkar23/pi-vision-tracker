# This script runs on the Raspberry Pi 4 and tests camera-driven eye movement through a PCA9685 board.
"""
Camera-driven eye mechanism test for Raspberry Pi 4 with PCA9685.
"""

# This import is used for timed delays, timestamps, and blink timing.
import time
# This import is used to allow a clean exit when the user presses Ctrl+C.
import sys
# This import is used for small helper math operations on servo values.
import math
# This import is used for optional type hints that make the script easier to maintain.
from typing import Dict, Optional, Tuple

# This line confirms that Python has started loading this v3 file.
print("eyes v3.py loaded: starting imports.", flush=True)

# This import is used to open the camera and perform face detection.
import cv2

# This line confirms that OpenCV imported correctly.
print("OpenCV imported.", flush=True)

# This import block enables the same MediaPipe face detector used by the older working camera script.
try:
    # This import is used for fast face detection that worked well in the serial Pico version.
    import mediapipe as mp
# This block lets the script fall back to OpenCV Haar detection if MediaPipe is not installed.
except ImportError:
    # This assignment marks MediaPipe as unavailable without stopping the whole program.
    mp = None

# This import is used to access the Raspberry Pi camera through the supported libcamera stack.
try:
    # This import is preferred on Raspberry Pi OS because it supports the OV5647 camera cleanly.
    from picamera2 import Picamera2
except ImportError:
    # This assignment allows the rest of the file to detect that Picamera2 is unavailable.
    Picamera2 = None

# This constant stores the primary I2C address for the eye PCA9685 board.
PCA9685_ADDRESS = 0x40
# This constant stores the standard servo PWM frequency used by the PCA9685 board.
PCA9685_FREQUENCY = 50
# This constant selects the camera width used for face tracking.
FRAME_WIDTH = 640
# This constant selects the camera height used for face tracking.
FRAME_HEIGHT = 480
# This constant enables or disables the OpenCV preview window during testing.
SHOW_PREVIEW = True
# This constant enables or disables the debug overlay at startup.
DEBUG_OVERLAY_ENABLED = True
# This constant enables or disables automatic blinking at startup.
AUTO_BLINK_ENABLED = True
# This constant selects the fixed servo command update rate.
SERVO_UPDATE_HZ = 30.0
# This constant defines how much horizontal error is ignored to reduce jitter.
DEADBAND_X_PIXELS = 8
# This constant defines how much vertical error is ignored to reduce jitter.
DEADBAND_Y_PIXELS = 12
# This constant limits the horizontal eye movement speed in degrees per camera loop.
LR_MAX_STEP_DEGREES = 2.0
# This constant limits the vertical eye movement speed in degrees per camera loop.
UD_MAX_STEP_DEGREES = 1.8
# This constant limits eyelid movement speed in degrees per camera loop.
LID_MAX_STEP_DEGREES = 6.0
# This constant ignores tiny servo target changes so the hardware does not chatter.
SERVO_WRITE_EPSILON_DEGREES = 0.35
# This constant copies the proportional movement gain from the working Pico follower code.
TRACKING_KP = 0.06
# This constant multiplies horizontal face error so left and right movement is clearly visible.
LR_ERROR_GAIN = 1.4
# This constant multiplies vertical face error so standing and sitting create clearer up and down eye movement.
UD_ERROR_GAIN = 1.8
# This constant holds the last eye position briefly when face detection drops for a few frames.
FACE_LOST_HOLD_SECONDS = 1.0
# This constant defines the shortest allowed gap between automatic blinks.
BLINK_INTERVAL_MIN = 2.5
# This constant defines the longest allowed gap between automatic blinks.
BLINK_INTERVAL_MAX = 5.0
# This constant defines how long the lids stay fully closed during a smooth non-blocking blink.
BLINK_HOLD_SECONDS = 0.10
# This constant defines the fallback frame read delay when no preview window is shown.
IDLE_LOOP_DELAY_SECONDS = 0.02
# This constant controls whether a visible servo sweep runs at startup after calibration.
STARTUP_SWEEP_ENABLED = False
# This constant controls how many gentle steps are used to enter the safe startup pose.
SAFETY_STARTUP_STEPS = 24
# This constant controls the pause between gentle startup steps.
SAFETY_STARTUP_STEP_DELAY_SECONDS = 0.025
# This constant controls how many degrees each keyboard arrow press moves an eye axis.
KEYBOARD_EYE_STEP_DEGREES = 4.0
# This constant controls how long keyboard movement temporarily overrides camera tracking.
KEYBOARD_OVERRIDE_SECONDS = 1.2
# This constant controls the smoothing amount for detected face center values.
FACE_CENTER_SMOOTHING = 0.25
# This constant stores how long to hold the last good face target after brief detection loss.
LAST_GOOD_FACE_HOLD_SECONDS = 0.45
# This constant stores the minimum face area percent accepted as a real tracking target.
MIN_TRACK_FACE_AREA_PERCENT = 0.35
# This constant stores the face area percent below which the person is considered far away.
FAR_FACE_AREA_PERCENT = 3.0
# This constant enables or disables automatic left-right eye searching when no person is locked.
AUTO_SEARCH_ENABLED = True
# This constant controls how long the robot waits after losing a face before search starts.
SEARCH_START_AFTER_SECONDS = 1.2
# This constant stores the left edge of the safe search sweep.
SEARCH_LR_MIN = 55.0
# This constant stores the right edge of the safe search sweep.
SEARCH_LR_MAX = 125.0
# This constant stores the vertical eye angle used while searching.
SEARCH_UD_ANGLE = 95.0
# This constant controls how many degrees each search step moves.
SEARCH_STEP_DEGREES = 0.8
# This constant controls how many stable face frames are needed before search locks on.
SEARCH_LOCK_REQUIRED_FRAMES = 3
# This constant controls how much runtime key presses change eye speed settings.
RUNTIME_SPEED_STEP_DEGREES = 0.2
# This constant controls how much runtime key presses change deadband settings.
RUNTIME_DEADBAND_STEP_PIXELS = 2
# This constant controls how much runtime key presses change face gain settings.
RUNTIME_GAIN_STEP = 0.1
# This set stores key codes that OpenCV can report for the left arrow key.
KEY_LEFT_CODES = {81, 2424832}
# This set stores key codes that OpenCV can report for the up arrow key.
KEY_UP_CODES = {82, 2490368}
# This set stores key codes that OpenCV can report for the right arrow key.
KEY_RIGHT_CODES = {83, 2555904}
# This set stores key codes that OpenCV can report for the down arrow key.
KEY_DOWN_CODES = {84, 2621440}
# This constant controls how often tracking values are printed in the terminal.
DEBUG_PRINT_INTERVAL_SECONDS = 1.0
# This constant selects the OpenCV camera index used when Picamera2 cannot open a CSI camera.
OPENCV_CAMERA_INDEX = 0
# This constant matches your working camera test by flipping both axes after capture.
CAMERA_FLIP_CODE = -1
# This constant mirrors the already-corrected camera frame horizontally when left and right need swapping.
MIRROR_FRAME_HORIZONTAL = True
# This constant enables the fast smoothing filter from your working camera setting.
CAMERA_GAUSSIAN_BLUR = True
# This constant keeps the brightness gain from your working camera setting.
CAMERA_BRIGHTNESS_ALPHA = 1.2
# This constant keeps the brightness offset from your working camera setting.
CAMERA_BRIGHTNESS_BETA = 15

# This dictionary maps your exact eye mechanism names to PCA9685 channel numbers.
SERVO_CHANNELS = {
    # This PCA9685 channel moves both eyes left and right as shown in the wiring image.
    "LR": 10,
    # This PCA9685 channel moves both eyes up and down as shown in the wiring image.
    "UD": 11,
    # This PCA9685 channel drives the top left eyelid as shown in the wiring image.
    "TL": 12,
    # This PCA9685 channel drives the bottom left eyelid as shown in the wiring image.
    "BL": 13,
    # This PCA9685 channel drives the top right eyelid as shown in the wiring image.
    "TR": 14,
    # This PCA9685 channel drives the bottom right eyelid as shown in the wiring image.
    "BR": 15,
}

# This dictionary stores the safe mechanical angle limits for each eye-related servo.
SERVO_LIMITS = {
    # This range matches your current horizontal eye sweep limits.
    "LR": (40.0, 140.0),
    # This range matches your current vertical eye sweep limits.
    "UD": (60.0, 140.0),
    # This range matches your current top left eyelid limits.
    "TL": (90.0, 160.0),
    # This range matches your current bottom left eyelid limits.
    "BL": (90.0, 30.0),
    # This range matches your current top right eyelid limits.
    "TR": (90.0, 30.0),
    # This range matches your current bottom right eyelid limits.
    "BR": (90.0, 140.0),
}

# This dictionary stores optional pulse tuning values for MG90S style servos on the PCA9685.
SERVO_PULSE_US = {
    # This minimum pulse width is a safe starting point for many small servos.
    "min_pulse": 500,
    # This maximum pulse width is a safe starting point for many small servos.
    "max_pulse": 2500,
}

# This dictionary stores the neutral eye position used at startup and during face loss.
NEUTRAL_POSITION = {
    # This neutral horizontal position centers the eyes.
    "LR": 90.0,
    # This neutral vertical position centers the eyes.
    "UD": 95.0,
}

# This dictionary stores the current eyelid trim settings for the open-eye pose.
EYELID_OPEN_LIMITS = {
    # This open value keeps the top left lid raised.
    "TL": 160.0,
    # This open value keeps the bottom left lid lowered.
    "BL": 30.0,
    # This open value keeps the top right lid raised.
    "TR": 30.0,
    # This open value keeps the bottom right lid lowered.
    "BR": 140.0,
}

# This dictionary stores the eyelid angles used when both top and bottom lids must close together.
EYELID_CLOSED_LIMITS = {
    # This closed value brings the top left eyelid down.
    "TL": 90.0,
    # This closed value brings the bottom left eyelid up.
    "BL": 90.0,
    # This closed value brings the top right eyelid down.
    "TR": 90.0,
    # This closed value brings the bottom right eyelid up.
    "BR": 90.0,
}


# This class stores runtime-tunable behavior without changing calibrated servo endpoints.
class RuntimeSettings:
    # This initializer copies the startup constants into mutable runtime values.
    def __init__(self) -> None:
        # This line stores whether the debug overlay should be drawn.
        self.debug_overlay_enabled = DEBUG_OVERLAY_ENABLED
        # This line stores whether automatic blinking should run.
        self.auto_blink_enabled = AUTO_BLINK_ENABLED
        # This line stores whether keyboard-only manual mode is active.
        self.manual_mode_enabled = False
        # This line stores whether automatic search mode is allowed to run.
        self.auto_search_enabled = AUTO_SEARCH_ENABLED
        # This line stores the horizontal movement speed limit.
        self.lr_max_step_degrees = LR_MAX_STEP_DEGREES
        # This line stores the vertical movement speed limit.
        self.ud_max_step_degrees = UD_MAX_STEP_DEGREES
        # This line stores the eyelid movement speed limit.
        self.lid_max_step_degrees = LID_MAX_STEP_DEGREES
        # This line stores the minimum interval between servo command writes.
        self.servo_update_interval_seconds = 1.0 / SERVO_UPDATE_HZ
        # This line stores the horizontal tracking deadband.
        self.deadband_x_pixels = DEADBAND_X_PIXELS
        # This line stores the vertical tracking deadband.
        self.deadband_y_pixels = DEADBAND_Y_PIXELS
        # This line stores the horizontal face-to-eye gain.
        self.lr_error_gain = LR_ERROR_GAIN
        # This line stores the vertical face-to-eye gain.
        self.ud_error_gain = UD_ERROR_GAIN
        # This line stores the face center smoothing amount.
        self.face_center_smoothing = FACE_CENTER_SMOOTHING
        # This line stores how long the last good face target is reused after brief detection loss.
        self.last_good_face_hold_seconds = LAST_GOOD_FACE_HOLD_SECONDS
        # This line stores the minimum face area percent accepted for tracking.
        self.min_track_face_area_percent = MIN_TRACK_FACE_AREA_PERCENT
        # This line stores the area percent below which the face is treated as far away.
        self.far_face_area_percent = FAR_FACE_AREA_PERCENT
        # This line stores the blink closed-hold duration.
        self.blink_hold_seconds = BLINK_HOLD_SECONDS
        # This line stores how many confident frames are required before locking from search mode.
        self.search_lock_required_frames = SEARCH_LOCK_REQUIRED_FRAMES
        # This line stores how fast the left-right search sweep moves.
        self.search_step_degrees = SEARCH_STEP_DEGREES


# This helper function clamps a value inside a safe minimum and maximum range.
def clamp(value: float, minimum: float, maximum: float) -> float:
    # This line returns the lower bound if the value is too small, the upper bound if too large, or the value itself.
    return max(minimum, min(maximum, value))


# This helper function blends the current angle toward a target angle to reduce jitter.
def smooth_move(current: float, target: float, amount: float) -> float:
    # This line performs a simple low-pass filter between the current and target values.
    return current + ((target - current) * amount)


# This helper function moves one angle toward another without exceeding a maximum step size.
def move_toward(current: float, target: float, max_step: float) -> float:
    # This line calculates how far the current angle is from the requested target.
    delta = target - current
    # This line returns the target directly when the remaining distance is already small enough.
    if abs(delta) <= max_step:
        # This line finishes the movement without overshooting the target.
        return target
    # This line moves upward by the maximum allowed step when the target is higher.
    if delta > 0:
        # This line returns the next safe upward angle.
        return current + max_step
    # This line returns the next safe downward angle when the target is lower.
    return current - max_step


# This helper function converts one numeric range into another numeric range.
def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    # This line prevents division by zero if the input range is invalid.
    if math.isclose(in_max, in_min):
        # This line returns the lower output value when the input range cannot be used.
        return out_min
    # This line scales the input value into the output range.
    return out_min + ((value - in_min) * (out_max - out_min) / (in_max - in_min))


# This helper function applies the same camera cleanup used by your working Picamera2 test.
def prepare_camera_frame(frame):
    # This line flips the frame in both directions to correct the mounted camera orientation.
    frame = cv2.flip(frame, CAMERA_FLIP_CODE)
    # This line mirrors the frame horizontally when the preview or tracking direction needs left-right correction.
    if MIRROR_FRAME_HORIZONTAL:
        # This line flips the frame left to right after the mounted-camera correction.
        frame = cv2.flip(frame, 1)
    # This line checks whether Picamera2 returned a four-channel frame.
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        # This line converts four-channel camera output into three-channel RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    # This branch handles normal three-channel camera output.
    else:
        # This line applies the same color conversion used in your working camera test.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # This line checks whether the lightweight smoothing filter is enabled.
    if CAMERA_GAUSSIAN_BLUR:
        # This line applies fast 3 by 3 Gaussian smoothing instead of heavy denoise.
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # This line applies the brightness and contrast correction from your working camera test.
    frame = cv2.convertScaleAbs(frame, alpha=CAMERA_BRIGHTNESS_ALPHA, beta=CAMERA_BRIGHTNESS_BETA)
    # This line returns the corrected frame to the tracking loop.
    return frame


# This helper function finds the largest detected face because that is usually the person nearest the robot.
def select_largest_face(faces) -> Optional[Tuple[int, int, int, int]]:
    # This line returns no face when the detector reports nothing.
    if len(faces) == 0:
        # This line signals that there is no target to follow.
        return None
    # This line selects the face rectangle with the largest area.
    return max(faces, key=lambda face: face[2] * face[3])


# This helper function estimates readable x, y, and z-style tracking values from the detected face.
def calculate_face_metrics(face: Optional[Tuple[int, int, int, int]]) -> Dict[str, float]:
    # This line returns zero values when no face is detected.
    if face is None:
        # This dictionary describes the no-face state for the overlay and terminal logs.
        return {"found": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0, "h": 0.0, "x_error": 0.0, "y_error": 0.0}
    # This line unpacks the face rectangle from OpenCV.
    x, y, w, h = face
    # This line calculates the center x coordinate of the face.
    center_x = x + (w / 2.0)
    # This line calculates the center y coordinate of the face.
    center_y = y + (h / 2.0)
    # This line calculates horizontal error using the same sign as CameraProcessor.py.
    x_error = (FRAME_WIDTH / 2.0) - center_x
    # This line calculates vertical error using the same sign as CameraProcessor.py.
    y_error = (FRAME_HEIGHT / 2.0) - center_y
    # This line estimates z as face-size proximity because a single camera cannot measure true depth.
    z_estimate = (w * h) / (FRAME_WIDTH * FRAME_HEIGHT) * 100.0
    # This dictionary returns all values needed for display and debugging.
    return {"found": 1.0, "x": center_x, "y": center_y, "z": z_estimate, "w": float(w), "h": float(h), "x_error": x_error, "y_error": y_error}


# This class wraps all eye servo behavior so the camera loop can stay clean.
class EyeMechanismController:
    # This initializer creates the PCA9685 connection and the servo objects for the eye mechanism.
    def __init__(self, settings: RuntimeSettings) -> None:
        # This line stores the runtime settings used for speed limits and blink timing.
        self.settings = settings
        # This line imports Raspberry Pi board pin names only when servo hardware is being initialized.
        import board
        # This line imports the Raspberry Pi I2C helper only when servo hardware is being initialized.
        import busio
        # This line imports the Adafruit servo helper only when servo hardware is being initialized.
        from adafruit_motor import servo
        # This line imports the PCA9685 driver only when servo hardware is being initialized.
        from adafruit_pca9685 import PCA9685
        # This line opens the Raspberry Pi I2C bus using the default board pins.
        self.i2c = busio.I2C(board.SCL, board.SDA)
        # This line creates the PCA9685 object at the configured eye-board address.
        self.pca = PCA9685(self.i2c, address=PCA9685_ADDRESS)
        # This line sets the PWM update rate to the standard servo frequency.
        self.pca.frequency = PCA9685_FREQUENCY
        # This line creates the dictionary that will hold the servo objects keyed by your exact names.
        self.servos: Dict[str, object] = {}
        # This line loops through each named servo channel in your current eye mechanism.
        for name, channel in SERVO_CHANNELS.items():
            # This line creates a Servo helper object on the correct PCA9685 channel.
            self.servos[name] = servo.Servo(
                # This line points the servo object at the selected PCA9685 output.
                self.pca.channels[channel],
                # This line applies the configured minimum pulse width.
                min_pulse=SERVO_PULSE_US["min_pulse"],
                # This line applies the configured maximum pulse width.
                max_pulse=SERVO_PULSE_US["max_pulse"],
            )
        # This line stores the current horizontal eye angle so motion can be smoothed.
        self.current_lr = NEUTRAL_POSITION["LR"]
        # This line stores the current vertical eye angle so motion can be smoothed.
        self.current_ud = NEUTRAL_POSITION["UD"]
        # This line stores the requested horizontal eye target separately from the current hardware angle.
        self.target_lr = NEUTRAL_POSITION["LR"]
        # This line stores the requested vertical eye target separately from the current hardware angle.
        self.target_ud = NEUTRAL_POSITION["UD"]
        # This dictionary stores the current eyelid angles used by the smooth lid animation.
        self.current_lids = dict(EYELID_CLOSED_LIMITS)
        # This dictionary stores the requested eyelid angles used by the smooth lid animation.
        self.target_lids = dict(EYELID_OPEN_LIMITS)
        # This dictionary remembers the last angle written to each servo to avoid tiny jitter writes.
        self.last_written_angles: Dict[str, Optional[float]] = {name: None for name in SERVO_CHANNELS}
        # This line stores the next time at which an automatic blink is allowed.
        self.next_blink_time = time.monotonic() + BLINK_INTERVAL_MIN
        # This line stores the blink state as idle, closing, hold, or opening.
        self.blink_state = "idle"
        # This line stores the time when the current blink hold phase should end.
        self.blink_hold_until = 0.0
        # This line stores the last time servo commands were allowed to update.
        self.last_servo_update_time = 0.0
        # This line sends the mechanism gently to a safe open neutral pose at startup.
        self.safety_startup()

    # This method writes a safe angle to one named servo.
    def set_servo_angle(self, name: str, angle: float) -> None:
        # This line reads the safe minimum and maximum angles for the selected servo.
        minimum, maximum = SERVO_LIMITS[name]
        # This line clamps the requested angle inside the safe mechanical range.
        safe_angle = clamp(angle, min(minimum, maximum), max(minimum, maximum))
        # This line reads the last angle written to this servo.
        last_angle = self.last_written_angles[name]
        # This line avoids rewriting tiny angle changes that can cause audible jitter.
        if last_angle is not None and abs(safe_angle - last_angle) < SERVO_WRITE_EPSILON_DEGREES:
            # This line exits because the servo is already holding the requested position closely enough.
            return
        # This line sends the clamped angle to the servo object.
        self.servos[name].angle = safe_angle
        # This line stores the clamped angle as the latest hardware command for this servo.
        self.last_written_angles[name] = safe_angle

    # This method moves every servo to a centered assembly-safe position.
    def calibrate(self) -> None:
        # This line loops through every named servo in the eye mechanism.
        for name in self.servos:
            # This line sends a 90-degree center command to each servo.
            self.set_servo_angle(name, 90.0)
            # This short pause reduces sudden current spikes during calibration.
            time.sleep(0.03)

    # This method gently moves the mechanism into the safe open neutral pose at startup.
    def safety_startup(self) -> None:
        # This line sets the target horizontal eye position to neutral.
        self.target_lr = NEUTRAL_POSITION["LR"]
        # This line sets the target vertical eye position to neutral.
        self.target_ud = NEUTRAL_POSITION["UD"]
        # This line sets the eyelid target to the calibrated open pose.
        self.target_lids = dict(EYELID_OPEN_LIMITS)
        # This line loops through a fixed number of small startup steps.
        for _ in range(SAFETY_STARTUP_STEPS):
            # This line advances eye and lid positions by one gentle step.
            self.update(force=True)
            # This short pause prevents a sudden current spike during startup.
            time.sleep(SAFETY_STARTUP_STEP_DELAY_SECONDS)

    # This method moves the eye mechanism target to the normal open and centered pose.
    def neutral(self) -> None:
        # This line sets the horizontal eye target to the calibrated center.
        self.target_lr = NEUTRAL_POSITION["LR"]
        # This line sets the vertical eye target to the calibrated center.
        self.target_ud = NEUTRAL_POSITION["UD"]
        # This line asks the lids to open independently from vertical eye movement.
        self.open_lids()
        # This line advances one smooth update toward the neutral pose.
        self.update(force=True)

    # This method smoothly returns the mechanism to neutral open pose before shutdown.
    def smooth_shutdown(self) -> None:
        # This line sets the horizontal target to the calibrated neutral angle.
        self.target_lr = NEUTRAL_POSITION["LR"]
        # This line sets the vertical target to the calibrated neutral angle.
        self.target_ud = NEUTRAL_POSITION["UD"]
        # This line sets the eyelid target to the calibrated open pose.
        self.target_lids = dict(EYELID_OPEN_LIMITS)
        # This line cancels any active blink state during shutdown.
        self.blink_state = "idle"
        # This line loops through enough fixed steps to reach the neutral open pose smoothly.
        for _ in range(SAFETY_STARTUP_STEPS):
            # This line advances one forced smooth step toward the shutdown pose.
            self.update(force=True)
            # This short pause keeps shutdown movement gentle.
            time.sleep(SAFETY_STARTUP_STEP_DELAY_SECONDS)

    # This method starts a smooth automatic blink without blocking camera tracking.
    def blink(self) -> None:
        # This line ignores a new blink request when a blink animation is already running.
        if self.blink_state != "idle":
            # This line exits so overlapping blink animations do not fight each other.
            return
        # This line starts the smooth closing phase of the blink.
        self.blink_state = "closing"
        # This line sets all eyelid targets to their calibrated closed positions.
        self.target_lids = dict(EYELID_CLOSED_LIMITS)

    # This method starts a smooth manual eyelid close without changing the eye direction.
    def close_lids(self) -> None:
        # This line cancels any active blink phase because the user requested manual closed lids.
        self.blink_state = "idle"
        # This line sets all eyelid targets to their calibrated closed positions.
        self.target_lids = dict(EYELID_CLOSED_LIMITS)

    # This method starts a smooth manual eyelid open without changing the eye direction.
    def open_lids(self) -> None:
        # This line cancels any active blink phase because the user requested manual open lids.
        self.blink_state = "idle"
        # This line sets all eyelid targets to their calibrated open positions.
        self.target_lids = dict(EYELID_OPEN_LIMITS)

    # This method performs a short visible servo sweep so you know the script is actually running.
    def startup_sweep(self) -> None:
        # This line moves the horizontal eye servo to the first end.
        self.set_servo_angle("LR", SERVO_LIMITS["LR"][0])
        # This line waits long enough for you to see the first LR position.
        time.sleep(0.45)
        # This line moves the horizontal eye servo to the second end.
        self.set_servo_angle("LR", SERVO_LIMITS["LR"][1])
        # This line waits long enough for you to see the second LR position.
        time.sleep(0.45)
        # This line returns the horizontal eye servo to neutral.
        self.set_servo_angle("LR", NEUTRAL_POSITION["LR"])
        # This line waits briefly before the vertical sweep starts.
        time.sleep(0.25)
        # This line moves the vertical eye servo to the first end with lid follow.
        self.control_ud_and_lids(SERVO_LIMITS["UD"][0])
        # This line waits long enough for you to see the first UD position.
        time.sleep(0.45)
        # This line moves the vertical eye servo to the second end with lid follow.
        self.control_ud_and_lids(SERVO_LIMITS["UD"][1])
        # This line waits long enough for you to see the second UD position.
        time.sleep(0.45)
        # This line returns the eyes and lids to the normal neutral pose.
        self.neutral()

    # This method updates only the vertical eye target so eyelids stay independent from eye-follow.
    def control_ud_and_lids(self, ud_angle: float) -> None:
        # This line stores the vertical target while leaving eyelid targets unchanged.
        self.target_ud = clamp(ud_angle, min(SERVO_LIMITS["UD"]), max(SERVO_LIMITS["UD"]))
        # This line advances one smooth update toward the requested vertical target.
        self.update()

    # This method updates the horizontal and vertical eye position together.
    def look_at(self, target_lr: float, target_ud: float) -> None:
        # This line stores the horizontal target inside the calibrated left-right range.
        self.target_lr = clamp(target_lr, min(SERVO_LIMITS["LR"]), max(SERVO_LIMITS["LR"]))
        # This line stores the vertical target inside the calibrated up-down range.
        self.target_ud = clamp(target_ud, min(SERVO_LIMITS["UD"]), max(SERVO_LIMITS["UD"]))
        # This line advances one smooth update toward the requested eye targets.
        self.update()

    # This method advances both eyes and eyelids by one smooth rate-limited step.
    def update(self, force: bool = False) -> None:
        # This line reads the current monotonic time for fixed-rate servo command timing.
        now = time.monotonic()
        # This line skips hardware writes until the configured servo command interval has passed.
        if not force and now - self.last_servo_update_time < self.settings.servo_update_interval_seconds:
            # This line exits early so the PCA9685 is not over-commanded.
            return
        # This line records that this loop is allowed to send servo commands.
        self.last_servo_update_time = now
        # This line advances the horizontal eye angle without exceeding the configured speed limit.
        self.current_lr = move_toward(self.current_lr, self.target_lr, self.settings.lr_max_step_degrees)
        # This line advances the vertical eye angle without exceeding the configured speed limit.
        self.current_ud = move_toward(self.current_ud, self.target_ud, self.settings.ud_max_step_degrees)
        # This line writes the rate-limited horizontal eye angle to the servo driver.
        self.set_servo_angle("LR", self.current_lr)
        # This line writes the rate-limited vertical eye angle to the servo driver.
        self.set_servo_angle("UD", self.current_ud)
        # This line advances the smooth blink or manual eyelid animation.
        self.update_lids()

    # This method advances all eyelid servos by one smooth animation step.
    def update_lids(self) -> None:
        # This line loops through each calibrated eyelid servo.
        for name in EYELID_OPEN_LIMITS:
            # This line moves the current lid angle toward the active lid target.
            self.current_lids[name] = move_toward(self.current_lids[name], self.target_lids[name], self.settings.lid_max_step_degrees)
            # This line writes the smooth eyelid angle to the servo driver.
            self.set_servo_angle(name, self.current_lids[name])
        # This line updates the blink state after the latest eyelid movement.
        self.update_blink_state()

    # This method moves the blink state machine between close, hold, and open phases.
    def update_blink_state(self) -> None:
        # This line reads the current monotonic time for blink hold timing.
        now = time.monotonic()
        # This line checks whether the blink is still closing.
        if self.blink_state == "closing":
            # This line checks whether every eyelid has reached its closed target.
            if all(math.isclose(self.current_lids[name], EYELID_CLOSED_LIMITS[name], abs_tol=0.6) for name in EYELID_CLOSED_LIMITS):
                # This line switches to the short closed-hold phase.
                self.blink_state = "hold"
                # This line stores when the closed-hold phase should finish.
                self.blink_hold_until = now + self.settings.blink_hold_seconds
        # This line checks whether the blink is holding closed long enough to be visible.
        elif self.blink_state == "hold":
            # This line checks whether the hold time has elapsed.
            if now >= self.blink_hold_until:
                # This line switches to the smooth opening phase.
                self.blink_state = "opening"
                # This line sets all eyelid targets back to the calibrated open pose.
                self.target_lids = dict(EYELID_OPEN_LIMITS)
        # This line checks whether the blink is opening.
        elif self.blink_state == "opening":
            # This line checks whether every eyelid has reached its open target.
            if all(math.isclose(self.current_lids[name], EYELID_OPEN_LIMITS[name], abs_tol=0.6) for name in EYELID_OPEN_LIMITS):
                # This line marks the blink animation as complete.
                self.blink_state = "idle"
                # This line schedules the next automatic blink after the completed animation.
                self.next_blink_time = now + BLINK_INTERVAL_MIN + ((BLINK_INTERVAL_MAX - BLINK_INTERVAL_MIN) * 0.5)

    # This method moves the eyes using the same incremental error behavior as the working Pico code.
    def track_error(self, error_x: float, error_y: float) -> None:
        # This line moves the horizontal eye servo using the same sign correction as FollowerBotPicoCode.py.
        self.current_lr = self._move_axis("LR", self.current_lr, error_x * -1.0, DEADBAND_X_PIXELS)
        # This line moves the vertical eye servo using the same direction as FollowerBotPicoCode.py.
        self.current_ud = self._move_axis("UD", self.current_ud, error_y, DEADBAND_Y_PIXELS)
        # This line writes the updated horizontal eye angle to the servo driver.
        self.set_servo_angle("LR", self.current_lr)
        # This line writes the updated vertical eye angle and matching eyelid follow movement.
        self.control_ud_and_lids(self.current_ud)

    # This helper moves one servo target by a proportional step outside the deadzone.
    def _move_axis(self, name: str, current_angle: float, error: float, deadzone: int) -> float:
        # This line exits without movement when the face error is inside the stable deadzone.
        if abs(error) <= deadzone:
            # This line returns the unchanged angle when movement is not needed.
            return current_angle
        # This line subtracts the deadzone from positive error so movement starts gently.
        if error > 0:
            # This line removes the ignored center area from positive motion.
            error = error - deadzone
        # This line adds the deadzone to negative error so movement starts gently.
        elif error < 0:
            # This line removes the ignored center area from negative motion.
            error = error + deadzone
        # This line converts the remaining pixel error into a servo angle step.
        step = int(TRACKING_KP * error)
        # This line guarantees at least one degree of movement when the error is outside the deadzone.
        if step == 0:
            # This line chooses the one-degree step direction from the error sign.
            step = 1 if error > 0 else -1
        # This line adds the calculated step to the current servo angle.
        new_angle = current_angle + step
        # This line reads the configured safe movement range for the selected servo.
        limit_a, limit_b = SERVO_LIMITS[name]
        # This line clamps the updated servo angle inside the safe movement range.
        return clamp(new_angle, min(limit_a, limit_b), max(limit_a, limit_b))

    # This method returns the eyes gently to neutral when no face is visible.
    def idle_return(self) -> None:
        # This line moves the eyes back toward their neutral horizontal position.
        self.look_at(NEUTRAL_POSITION["LR"], NEUTRAL_POSITION["UD"])

    # This method performs a scheduled blink when enough time has passed.
    def maybe_blink(self) -> None:
        # This line checks the current monotonic time for blink scheduling.
        now = time.monotonic()
        # This line exits when a blink animation is already running.
        if self.blink_state != "idle":
            # This line advances the active blink even if the eyes are currently holding still.
            self.update(force=True)
            # This line avoids starting a second blink before the first one finishes.
            return
        # This line returns immediately when the next blink time has not been reached.
        if now < self.next_blink_time:
            # This line exits the method without blinking yet.
            return
        # This line performs the blink animation.
        self.blink()
        # This line advances the first blink step immediately so the response feels direct.
        self.update(force=True)

    # This method releases the PCA9685 hardware cleanly on exit.
    def close(self) -> None:
        # This line deinitializes the PCA9685 object and releases the I2C peripheral.
        self.pca.deinit()


# This class wraps camera startup so the rest of the code can work with one simple interface.
class CameraReader:
    # This initializer opens the preferred Pi camera interface and falls back to OpenCV if needed.
    def __init__(self) -> None:
        # This line stores the Picamera2 object when the Pi camera stack is available.
        self.picam2 = None
        # This line stores the OpenCV capture object for fallback camera access.
        self.capture = None
        # This line marks whether the active camera source is Picamera2.
        self.using_picamera2 = False
        # This line starts the camera using the best supported method.
        self._open_camera()

    # This private helper tries Picamera2 first because it is the correct API for the OV5647 on Pi OS.
    def _open_camera(self) -> None:
        # This line checks if the Picamera2 module was imported successfully.
        if Picamera2 is not None:
            # This block catches cases where Picamera2 is installed but no CSI camera is detected.
            try:
                # This line confirms the script is about to open the camera through Picamera2.
                print("Opening camera with Picamera2.", flush=True)
                # This line creates a Picamera2 camera object.
                self.picam2 = Picamera2()
                # This line configures a preview stream at the requested resolution without changing your camera look.
                preview_config = self.picam2.create_preview_configuration(
                    # This line defines the output frame size used by the working camera test.
                    main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
                )
                # This line applies the preview configuration to the camera.
                self.picam2.configure(preview_config)
                # This line starts the camera stream.
                self.picam2.start()
                # This line applies the same automatic exposure and white-balance choices from your working camera test.
                self.picam2.set_controls({
                    # This line keeps automatic exposure enabled so lighting changes are handled by the camera.
                    "AeEnable": True,
                    # This line disables automatic white balance to keep the color response stable.
                    "AwbEnable": False,
                    # This line selects the same white-balance mode used in your working snippet.
                    "AwbMode": 3,
                })
                # This short pause gives the sensor time to settle before the first frame.
                time.sleep(1.0)
                # This line records that Picamera2 is the active camera source.
                self.using_picamera2 = True
                # This line confirms Picamera2 opened successfully.
                print("Picamera2 camera stream started.", flush=True)
                # This line exits after the preferred camera path has started successfully.
                return
            # This block handles Picamera2 startup failures and lets OpenCV try the camera next.
            except Exception as error:
                # This line clears the broken Picamera2 object before trying the fallback camera path.
                self.picam2 = None
                # This line reports the Picamera2 failure while keeping the program running.
                print(f"Picamera2 could not open a camera: {error}", flush=True)
        # This line confirms the script is about to try the OpenCV camera fallback.
        print(f"Opening camera with OpenCV VideoCapture({OPENCV_CAMERA_INDEX}).", flush=True)
        # This line falls back to OpenCV camera capture when Picamera2 is not present.
        self.capture = cv2.VideoCapture(OPENCV_CAMERA_INDEX)
        # This line sets the fallback camera width.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        # This line sets the fallback camera height.
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # This line raises an error when no camera source could be opened.
        if not self.capture.isOpened():
            # This error message tells you to install Picamera2 or check the camera device path.
            raise RuntimeError("Could not open camera. Check the CSI ribbon, camera enablement, or OpenCV camera index.")
        # This line confirms the fallback camera opened successfully.
        print("OpenCV fallback camera opened.", flush=True)

    # This method reads one frame from the active camera source.
    def read(self):
        # This line reads a frame from Picamera2 when that is the active camera path.
        if self.using_picamera2:
            # This line captures the latest frame as a NumPy array from Picamera2.
            frame = self.picam2.capture_array()
            # This line returns success and the raw frame for shared preprocessing.
            return True, frame
        # This line reads a frame from the OpenCV capture object in fallback mode.
        return self.capture.read()

    # This method releases the active camera resource during shutdown.
    def close(self) -> None:
        # This line stops the Picamera2 stream if it was started.
        if self.picam2 is not None:
            # This line stops the Pi camera cleanly.
            self.picam2.stop()
        # This line releases the OpenCV capture device if the fallback path was used.
        if self.capture is not None:
            # This line closes the fallback camera device.
            self.capture.release()


# This function loads the standard OpenCV face detector that ships with the package.
def create_face_detector():
    # This line prefers MediaPipe because it matched your older working camera tracker.
    if mp is not None:
        # This line creates a longer-range MediaPipe face detector so smaller distant faces can be found.
        detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)
        # This line returns the detector type and object together so the main loop can use the right API.
        return "mediapipe", detector
    # This line builds the full path to the bundled frontal face cascade file.
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    # This line creates the OpenCV cascade classifier from the standard file.
    detector = cv2.CascadeClassifier(cascade_path)
    # This line raises an error if the cascade file was not loaded correctly.
    if detector.empty():
        # This error helps catch broken OpenCV installations early.
        raise RuntimeError(f"Could not load face detector from {cascade_path}")
    # This line returns the fallback detector type and object to the main loop.
    return "haar", detector


# This function detects the best face rectangle using MediaPipe first or Haar as a fallback.
def detect_target_face(frame, detector_info) -> Optional[Tuple[int, int, int, int]]:
    # This line unpacks the detector type and detector object.
    detector_type, detector = detector_info
    # This line uses the MediaPipe detector when it is available.
    if detector_type == "mediapipe":
        # This line runs MediaPipe on the prepared RGB-style frame.
        results = detector.process(frame)
        # This line returns no face when MediaPipe does not report detections.
        if not results.detections:
            # This line signals that there is no valid target in this frame.
            return None
        # This line stores the best rectangle found so far.
        best_face = None
        # This line stores the largest face area found so far.
        best_area = 0
        # This line loops through every MediaPipe detection.
        for detection in results.detections:
            # This line reads the relative bounding box from the detection.
            bbox = detection.location_data.relative_bounding_box
            # This line converts the relative x coordinate into pixels.
            x = int(bbox.xmin * FRAME_WIDTH)
            # This line converts the relative y coordinate into pixels.
            y = int(bbox.ymin * FRAME_HEIGHT)
            # This line converts the relative width into pixels.
            w = int(bbox.width * FRAME_WIDTH)
            # This line converts the relative height into pixels.
            h = int(bbox.height * FRAME_HEIGHT)
            # This line clips the x coordinate to the visible frame.
            x = max(0, min(FRAME_WIDTH - 1, x))
            # This line clips the y coordinate to the visible frame.
            y = max(0, min(FRAME_HEIGHT - 1, y))
            # This line clips the width so the rectangle stays inside the frame.
            w = max(1, min(FRAME_WIDTH - x, w))
            # This line clips the height so the rectangle stays inside the frame.
            h = max(1, min(FRAME_HEIGHT - y, h))
            # This line calculates the face rectangle area.
            area = w * h
            # This line keeps the largest face as the tracking target.
            if area > best_area:
                # This line stores the largest face rectangle.
                best_face = (x, y, w, h)
                # This line stores the largest face area.
                best_area = area
        # This line returns the largest MediaPipe face rectangle.
        return best_face
    # This line converts the prepared frame to grayscale for the OpenCV fallback detector.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # This line detects frontal faces using the OpenCV Haar cascade fallback.
    faces = detector.detectMultiScale(
        # This line supplies the grayscale frame to the fallback detector.
        gray,
        # This line keeps the fallback detector reasonably responsive.
        scaleFactor=1.2,
        # This line requires repeated detections before accepting a face.
        minNeighbors=5,
        # This line allows smaller face boxes so farther people have a chance to be detected.
        minSize=(40, 40),
    )
    # This line returns the largest OpenCV face rectangle.
    return select_largest_face(faces)


# This function maps a detected face center in the image to eye servo target angles.
def face_to_eye_targets(face_center_x: float, face_center_y: float, settings: RuntimeSettings) -> Tuple[float, float]:
    # This line calculates horizontal error with the same sign used by CameraProcessor.py.
    x_error = (FRAME_WIDTH / 2.0) - face_center_x
    # This line calculates vertical error with the same sign used by CameraProcessor.py.
    y_error = (FRAME_HEIGHT / 2.0) - face_center_y
    # This line removes tiny horizontal camera noise so the eyes stay steadier.
    if abs(x_error) < settings.deadband_x_pixels:
        # This line zeroes the horizontal error inside the deadband.
        x_error = 0.0
    # This line removes tiny vertical camera noise so the eyelids do not chatter.
    if abs(y_error) < settings.deadband_y_pixels:
        # This line zeroes the vertical error inside the deadband.
        y_error = 0.0
    # This line boosts horizontal face error so left and right eye movement is easier to see.
    x_error = clamp(x_error * settings.lr_error_gain, -(FRAME_WIDTH / 2.0), FRAME_WIDTH / 2.0)
    # This line maps horizontal face error using the same left-right sign correction as the Pico code.
    target_lr = remap(x_error * -1.0, -(FRAME_WIDTH / 2.0), FRAME_WIDTH / 2.0, SERVO_LIMITS["LR"][0], SERVO_LIMITS["LR"][1])
    # This line boosts vertical face error so standing and sitting create a visible up or down command.
    y_error = clamp(y_error * settings.ud_error_gain, -(FRAME_HEIGHT / 2.0), FRAME_HEIGHT / 2.0)
    # This line maps vertical face error using the same up-down sign as the Pico code.
    target_ud = remap(y_error, -(FRAME_HEIGHT / 2.0), FRAME_HEIGHT / 2.0, SERVO_LIMITS["UD"][0], SERVO_LIMITS["UD"][1])
    # This line returns the horizontal and vertical target angles.
    return target_lr, target_ud


# This function draws simple debug graphics so you can see what the tracker is doing during testing.
def draw_debug(
    frame,
    face: Optional[Tuple[int, int, int, int]],
    metrics: Dict[str, float],
    target_lr: float,
    target_ud: float,
    servo_enabled: bool,
    settings: RuntimeSettings,
    tracking_status: str,
) -> None:
    # This line calculates the visual center of the frame for reference.
    frame_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
    # This line draws a small green marker at the center of the camera image.
    cv2.circle(frame, frame_center, 4, (0, 255, 0), -1)
    # This line draws a vertical center guide through the camera frame.
    cv2.line(frame, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (70, 70, 70), 1)
    # This line draws a horizontal center guide through the camera frame.
    cv2.line(frame, (0, FRAME_HEIGHT // 2), (FRAME_WIDTH, FRAME_HEIGHT // 2), (70, 70, 70), 1)
    # This line builds the servo status text for the overlay.
    servo_status = "ON" if servo_enabled else "OFF"
    # This line shows whether hardware servo control is active.
    cv2.putText(frame, f"Servo: {servo_status}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    # This line shows the current target servo angles on the preview.
    cv2.putText(frame, f"Target LR:{target_lr:6.1f}  UD:{target_ud:6.1f}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    # This line shows the estimated x, y, and z values on the preview.
    cv2.putText(frame, f"X:{metrics['x']:6.1f}  Y:{metrics['y']:6.1f}  Z:{metrics['z']:5.1f}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    # This line shows the face-center error values on the preview.
    cv2.putText(frame, f"Err X:{metrics['x_error']:6.1f}  Err Y:{metrics['y_error']:6.1f}", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    # This line shows live mode, blink, tuning, and far-person status.
    cv2.putText(frame, f"{tracking_status}  M:{settings.manual_mode_enabled} A:{settings.auto_blink_enabled} S:{settings.auto_search_enabled} D:{settings.debug_overlay_enabled}", (12, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # This line shows the current runtime tuning values.
    cv2.putText(frame, f"spd LR:{settings.lr_max_step_degrees:.1f} UD:{settings.ud_max_step_degrees:.1f} db:{settings.deadband_x_pixels}/{settings.deadband_y_pixels} gain:{settings.lr_error_gain:.1f}/{settings.ud_error_gain:.1f}", (12, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    # This line checks whether a face was detected in the current frame.
    if face is None:
        # This line writes a clear no-face message on the preview.
        cv2.putText(frame, "NO FACE FOUND", (12, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # This line exits without drawing a face box when no target exists.
        return
    # This line unpacks the detected face rectangle.
    x, y, w, h = face
    # This line draws a blue rectangle around the detected face.
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 2)
    # This line calculates the center point of the detected face.
    face_center = (x + (w // 2), y + (h // 2))
    # This line draws a red marker at the center of the detected face.
    cv2.circle(frame, face_center, 5, (0, 0, 255), -1)
    # This line draws a line from the frame center to the face center.
    cv2.line(frame, frame_center, face_center, (0, 255, 255), 1)
    # This line shows the face box size used to estimate z.
    cv2.putText(frame, f"Face W:{w} H:{h}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)


# This function applies runtime tuning keys and returns a readable status message.
def apply_runtime_tuning_key(key: int, settings: RuntimeSettings) -> Optional[str]:
    # This line decreases the horizontal eye movement speed when key 1 is pressed.
    if key == ord("1"):
        # This line clamps the horizontal speed so it cannot become too slow or negative.
        settings.lr_max_step_degrees = clamp(settings.lr_max_step_degrees - RUNTIME_SPEED_STEP_DEGREES, 0.4, 8.0)
        # This line returns a message for the terminal.
        return f"LR speed set to {settings.lr_max_step_degrees:.1f}"
    # This line increases the horizontal eye movement speed when key 2 is pressed.
    if key == ord("2"):
        # This line clamps the horizontal speed inside a safe tuning range.
        settings.lr_max_step_degrees = clamp(settings.lr_max_step_degrees + RUNTIME_SPEED_STEP_DEGREES, 0.4, 8.0)
        # This line returns a message for the terminal.
        return f"LR speed set to {settings.lr_max_step_degrees:.1f}"
    # This line decreases the vertical eye movement speed when key 3 is pressed.
    if key == ord("3"):
        # This line clamps the vertical speed so it cannot become too slow or negative.
        settings.ud_max_step_degrees = clamp(settings.ud_max_step_degrees - RUNTIME_SPEED_STEP_DEGREES, 0.4, 8.0)
        # This line returns a message for the terminal.
        return f"UD speed set to {settings.ud_max_step_degrees:.1f}"
    # This line increases the vertical eye movement speed when key 4 is pressed.
    if key == ord("4"):
        # This line clamps the vertical speed inside a safe tuning range.
        settings.ud_max_step_degrees = clamp(settings.ud_max_step_degrees + RUNTIME_SPEED_STEP_DEGREES, 0.4, 8.0)
        # This line returns a message for the terminal.
        return f"UD speed set to {settings.ud_max_step_degrees:.1f}"
    # This line decreases both tracking deadbands when key 5 is pressed.
    if key == ord("5"):
        # This line lowers the horizontal deadband while keeping a small noise buffer.
        settings.deadband_x_pixels = int(clamp(settings.deadband_x_pixels - RUNTIME_DEADBAND_STEP_PIXELS, 2, 60))
        # This line lowers the vertical deadband while keeping a small noise buffer.
        settings.deadband_y_pixels = int(clamp(settings.deadband_y_pixels - RUNTIME_DEADBAND_STEP_PIXELS, 2, 60))
        # This line returns a message for the terminal.
        return f"Deadband set to {settings.deadband_x_pixels}/{settings.deadband_y_pixels}"
    # This line increases both tracking deadbands when key 6 is pressed.
    if key == ord("6"):
        # This line raises the horizontal deadband inside a practical range.
        settings.deadband_x_pixels = int(clamp(settings.deadband_x_pixels + RUNTIME_DEADBAND_STEP_PIXELS, 2, 60))
        # This line raises the vertical deadband inside a practical range.
        settings.deadband_y_pixels = int(clamp(settings.deadband_y_pixels + RUNTIME_DEADBAND_STEP_PIXELS, 2, 60))
        # This line returns a message for the terminal.
        return f"Deadband set to {settings.deadband_x_pixels}/{settings.deadband_y_pixels}"
    # This line decreases horizontal face gain when key 7 is pressed.
    if key == ord("7"):
        # This line clamps horizontal gain to avoid making tracking unstable.
        settings.lr_error_gain = clamp(settings.lr_error_gain - RUNTIME_GAIN_STEP, 0.4, 4.0)
        # This line returns a message for the terminal.
        return f"LR gain set to {settings.lr_error_gain:.1f}"
    # This line increases horizontal face gain when key 8 is pressed.
    if key == ord("8"):
        # This line clamps horizontal gain to avoid making tracking unstable.
        settings.lr_error_gain = clamp(settings.lr_error_gain + RUNTIME_GAIN_STEP, 0.4, 4.0)
        # This line returns a message for the terminal.
        return f"LR gain set to {settings.lr_error_gain:.1f}"
    # This line decreases vertical face gain when key 9 is pressed.
    if key == ord("9"):
        # This line clamps vertical gain to avoid making tracking unstable.
        settings.ud_error_gain = clamp(settings.ud_error_gain - RUNTIME_GAIN_STEP, 0.4, 4.0)
        # This line returns a message for the terminal.
        return f"UD gain set to {settings.ud_error_gain:.1f}"
    # This line increases vertical face gain when key 0 is pressed.
    if key == ord("0"):
        # This line clamps vertical gain to avoid making tracking unstable.
        settings.ud_error_gain = clamp(settings.ud_error_gain + RUNTIME_GAIN_STEP, 0.4, 4.0)
        # This line returns a message for the terminal.
        return f"UD gain set to {settings.ud_error_gain:.1f}"
    # This line toggles keyboard-only manual mode when key m is pressed.
    if key == ord("m"):
        # This line flips manual mode between camera tracking and keyboard-only control.
        settings.manual_mode_enabled = not settings.manual_mode_enabled
        # This line returns a message for the terminal.
        return f"Manual mode set to {settings.manual_mode_enabled}"
    # This line toggles automatic blinking when key a is pressed.
    if key == ord("a"):
        # This line flips automatic blink scheduling on or off.
        settings.auto_blink_enabled = not settings.auto_blink_enabled
        # This line returns a message for the terminal.
        return f"Auto blink set to {settings.auto_blink_enabled}"
    # This line toggles debug overlay drawing when key d is pressed.
    if key == ord("d"):
        # This line flips the debug overlay on or off.
        settings.debug_overlay_enabled = not settings.debug_overlay_enabled
        # This line returns a message for the terminal.
        return f"Debug overlay set to {settings.debug_overlay_enabled}"
    # This line toggles automatic left-right search when key s is pressed.
    if key == ord("s"):
        # This line flips search mode between enabled and disabled.
        settings.auto_search_enabled = not settings.auto_search_enabled
        # This line returns a message for the terminal.
        return f"Auto search set to {settings.auto_search_enabled}"
    # This line reports that the key was not a tuning key.
    return None


# This function contains the main camera and servo test loop.
def main() -> int:
    # This line prints a startup message so the terminal shows that the script has begun.
    print("Starting eye mechanism camera test.", flush=True)
    # This line creates the mutable runtime settings for v3 tuning.
    settings = RuntimeSettings()
    # This line creates a placeholder so the camera can still run if servo hardware fails.
    eye_controller = None
    # This line tries to create the eye servo controller for the PCA9685 board.
    try:
        # This line creates the eye servo controller for the PCA9685 board.
        eye_controller = EyeMechanismController(settings)
        # This line prints a message after the PCA9685 and servo objects are ready.
        print("Eye controller initialized.", flush=True)
        # This line runs a visible hardware sweep when startup sweep is enabled.
        if STARTUP_SWEEP_ENABLED:
            # This line prints a message before the sweep begins.
            print("Running startup servo sweep.", flush=True)
            # This line performs the visible left-right and up-down sweep.
            eye_controller.startup_sweep()
            # This line prints a message after the sweep finishes.
            print("Startup servo sweep completed.", flush=True)
    # This block keeps the camera preview alive even when the PCA9685 setup fails.
    except Exception as error:
        # This line prints the servo hardware error clearly.
        print(f"Servo control disabled because PCA9685 setup failed: {error}", flush=True)
        # This line tells the user the camera view will still be shown.
        print("Continuing in camera-only debug mode.", flush=True)
    # This line opens the camera using Picamera2 or the fallback OpenCV path.
    camera = CameraReader()
    # This line prints a message after the camera opens successfully.
    print("Camera initialized.", flush=True)
    # This line loads the standard OpenCV frontal face detector.
    detector = create_face_detector()
    # This line prints a message after the face detector is ready.
    print("Face detector initialized. Press q in the preview window to stop.", flush=True)
    # This line creates the OpenCV preview window only when preview output is enabled.
    if SHOW_PREVIEW:
        # This line creates the OpenCV preview window before the loop starts.
        cv2.namedWindow("Eye Mechanism Camera Test", cv2.WINDOW_NORMAL)
        # This line sets the preview window size so it is visible on the desktop.
        cv2.resizeWindow("Eye Mechanism Camera Test", FRAME_WIDTH, FRAME_HEIGHT)
        # This line moves the preview window near the top-left corner of the desktop.
        cv2.moveWindow("Eye Mechanism Camera Test", 40, 40)
        # This line prints a message after the preview window is created.
        print("Preview window created. Keys: arrows move, b blink, o open, c close, m manual, a auto-blink, s auto-search, d overlay, 1-0 tune, q/Esc quit.", flush=True)
    # This line tracks whether a clean shutdown has already been started.
    running = True
    # This line stores the last time debug values were printed to the terminal.
    last_debug_print = 0.0
    # This line stores the latest LR target for overlay display.
    target_lr = NEUTRAL_POSITION["LR"]
    # This line stores the latest UD target for overlay display.
    target_ud = NEUTRAL_POSITION["UD"]
    # This line stores the last time a valid face was seen.
    last_face_seen_time = 0.0
    # This line stores when keyboard eye control should stop overriding camera tracking.
    manual_override_until = 0.0
    # This line stores the smoothed face center x coordinate.
    smoothed_face_center_x = None
    # This line stores the smoothed face center y coordinate.
    smoothed_face_center_y = None
    # This line stores the last good tracking target before brief face loss.
    last_good_target = (target_lr, target_ud)
    # This line stores the last time a valid confident face was accepted.
    last_good_face_time = 0.0
    # This line stores the current tracking status for logs and overlay.
    tracking_status = "STARTING"
    # This line tracks whether the automatic left-right search sweep is currently active.
    search_mode_active = False
    # This line stores the current direction of the automatic left-right search sweep.
    search_direction = 1.0
    # This line stores the current target angle used by automatic search mode.
    search_lr_target = NEUTRAL_POSITION["LR"]
    # This line counts stable confident face frames before search mode locks onto a target.
    stable_face_frames = 0
    # This line starts the main test block that guarantees cleanup on exit.
    try:
        # This line runs until the user quits the preview or presses Ctrl+C.
        while running:
            # This line reads one frame from the active camera source.
            success, frame = camera.read()
            # This line skips the rest of the loop when a frame could not be read.
            if not success:
                # This short pause prevents a tight error loop when the camera stalls.
                time.sleep(IDLE_LOOP_DELAY_SECONDS)
                # This line continues to the next loop iteration.
                continue
            # This line applies the working camera flip, color, smoothing, and brightness settings.
            frame = prepare_camera_frame(frame)
            # This line detects the largest face using MediaPipe when available.
            target_face = detect_target_face(frame, detector)
            # This line calculates display metrics for the current target.
            metrics = calculate_face_metrics(target_face)
            # This line reads the current monotonic time for face-hold and debug timing.
            now = time.monotonic()
            # This line checks whether keyboard eye control is temporarily overriding camera tracking.
            manual_override_active = now < manual_override_until
            # This line checks whether the detected face is large enough to trust for tracking.
            face_is_confident = target_face is not None and metrics["z"] >= settings.min_track_face_area_percent
            # This line checks whether a detected person is far away but still visible enough to report.
            person_is_far = target_face is not None and metrics["z"] < settings.far_face_area_percent
            # This line counts consecutive confident face frames so search mode locks only after stable detection.
            if face_is_confident:
                # This line records one more stable face frame.
                stable_face_frames += 1
            # This branch resets the stable face counter when the current frame is not confident.
            else:
                # This line clears the lock counter because the face target is missing or too weak.
                stable_face_frames = 0
            # This line updates the status when a face exists but is too small for confident tracking.
            if target_face is not None and not face_is_confident:
                # This line labels the detection as too far or too weak for stable tracking.
                tracking_status = "FACE TOO FAR / WEAK"
            # This line updates the hardware toward the keyboard target while manual override or manual mode is active.
            if (manual_override_active or settings.manual_mode_enabled) and eye_controller is not None:
                # This line cancels automatic search while the user is manually controlling the eyes.
                search_mode_active = False
                # This line keeps moving smoothly toward the latest keyboard-selected eye target.
                eye_controller.look_at(target_lr, target_ud)
                # This line stores the current horizontal angle for the debug overlay.
                target_lr = eye_controller.current_lr
                # This line stores the current vertical angle for the debug overlay.
                target_ud = eye_controller.current_ud
                # This line marks the current status as manual control.
                tracking_status = "MANUAL MODE" if settings.manual_mode_enabled else "KEYBOARD OVERRIDE"
            # This line holds search mode briefly while a newly found face proves it is stable.
            elif search_mode_active and face_is_confident and stable_face_frames < settings.search_lock_required_frames:
                # This line records that a possible target is visible during search mode.
                last_face_seen_time = now
                # This line records that a possible confident target is being checked for lock-on.
                last_good_face_time = now
                # This line marks the status as a pending target lock instead of full tracking.
                tracking_status = "LOCKING TARGET"
                # This line keeps the eyes at the current search position while the lock counter fills.
                if eye_controller is not None:
                    # This line continues sending the current search target at the safe servo update rate.
                    eye_controller.look_at(target_lr, target_ud)
                    # This line stores the current horizontal angle for the debug overlay.
                    target_lr = eye_controller.current_lr
                    # This line stores the current vertical angle for the debug overlay.
                    target_ud = eye_controller.current_ud
            # This line checks whether a valid confident face was found and camera control is allowed.
            elif face_is_confident:
                # This line checks whether tracking is taking over from automatic search mode.
                was_searching = search_mode_active
                # This line turns off automatic search because a stable target is ready to track.
                search_mode_active = False
                # This line records that a valid face was found in this loop.
                last_face_seen_time = now
                # This line records that a confident face target was accepted in this loop.
                last_good_face_time = now
                # This line unpacks the target face rectangle.
                x, y, w, h = target_face
                # This line calculates the center x position of the target face.
                face_center_x = x + (w / 2.0)
                # This line calculates the center y position of the target face.
                face_center_y = y + (h / 2.0)
                # This line resets smoothing when search mode first locks onto a target.
                if was_searching:
                    # This line starts the smoothed x value at the locked target center.
                    smoothed_face_center_x = face_center_x
                    # This line starts the smoothed y value at the locked target center.
                    smoothed_face_center_y = face_center_y
                    # This line clears the lock counter now that search has handed over to tracking.
                    stable_face_frames = 0
                # This line initializes face smoothing from the first confident face center.
                if smoothed_face_center_x is None or smoothed_face_center_y is None:
                    # This line stores the first confident face center directly.
                    smoothed_face_center_x = face_center_x
                    # This line stores the first confident face center directly.
                    smoothed_face_center_y = face_center_y
                # This line smooths face center x before it is converted into a servo target.
                smoothed_face_center_x = smooth_move(smoothed_face_center_x, face_center_x, settings.face_center_smoothing)
                # This line smooths face center y before it is converted into a servo target.
                smoothed_face_center_y = smooth_move(smoothed_face_center_y, face_center_y, settings.face_center_smoothing)
                # This line calculates horizontal error exactly like CameraProcessor.py.
                error_x = (FRAME_WIDTH / 2.0) - smoothed_face_center_x
                # This line calculates vertical error exactly like CameraProcessor.py.
                error_y = (FRAME_HEIGHT / 2.0) - smoothed_face_center_y
                # This line converts the face position directly into the desired eye angles.
                target_lr, target_ud = face_to_eye_targets(smoothed_face_center_x, smoothed_face_center_y, settings)
                # This line stores the latest confident target for brief face-loss hold behavior.
                last_good_target = (target_lr, target_ud)
                # This line stores a clear status when the person is visible but far away.
                tracking_status = "PERSON FAR AWAY" if person_is_far else "TRACKING FACE"
                # This line moves the eye servos directly toward the face target when hardware is active.
                if eye_controller is not None:
                    # This line writes a fast smoothed LR and UD update so the eyes move parallel with the face.
                    eye_controller.look_at(target_lr, target_ud)
                    # This line stores the current horizontal angle for the debug overlay.
                    target_lr = eye_controller.current_lr
                    # This line stores the current vertical angle for the debug overlay.
                    target_ud = eye_controller.current_ud
            # This line holds the last good face target through brief detection drops.
            elif not search_mode_active and now - last_good_face_time <= settings.last_good_face_hold_seconds:
                # This line reuses the most recent confident tracking target.
                target_lr, target_ud = last_good_target
                # This line labels the status as last-target hold.
                tracking_status = "HOLDING LAST FACE"
                # This line keeps the eyes moving smoothly toward the held target.
                if eye_controller is not None:
                    # This line writes the held face target at the configured servo update rate.
                    eye_controller.look_at(target_lr, target_ud)
                    # This line stores the current horizontal angle for the debug overlay.
                    target_lr = eye_controller.current_lr
                    # This line stores the current vertical angle for the debug overlay.
                    target_ud = eye_controller.current_ud
            else:
                # This line checks how long the face has been missing.
                face_missing_seconds = now - last_face_seen_time
                # This line waits briefly before returning or keeps searching if search mode is already active.
                if search_mode_active or face_missing_seconds > FACE_LOST_HOLD_SECONDS:
                    # This line checks whether automatic search should start after the missing-face delay.
                    if settings.auto_search_enabled and (search_mode_active or face_missing_seconds > SEARCH_START_AFTER_SECONDS):
                        # This line initializes search mode on the first search frame.
                        if not search_mode_active:
                            # This line marks automatic search as active.
                            search_mode_active = True
                            # This line starts the sweep from the current displayed horizontal target.
                            search_lr_target = target_lr
                            # This line chooses the first sweep direction based on the current side of neutral.
                            search_direction = 1.0 if target_lr <= NEUTRAL_POSITION["LR"] else -1.0
                        # This line advances the search target by one small left-right step.
                        search_lr_target += settings.search_step_degrees * search_direction
                        # This line checks whether the search target reached the right sweep limit.
                        if search_lr_target >= SEARCH_LR_MAX:
                            # This line clamps the search target to the safe right sweep limit.
                            search_lr_target = SEARCH_LR_MAX
                            # This line reverses search direction back toward the left.
                            search_direction = -1.0
                        # This line checks whether the search target reached the left sweep limit.
                        elif search_lr_target <= SEARCH_LR_MIN:
                            # This line clamps the search target to the safe left sweep limit.
                            search_lr_target = SEARCH_LR_MIN
                            # This line reverses search direction back toward the right.
                            search_direction = 1.0
                        # This line stores the current search horizontal target for servo movement and overlay.
                        target_lr = search_lr_target
                        # This line keeps the eyes level while the robot searches left and right.
                        target_ud = SEARCH_UD_ANGLE
                        # This line keeps the eyes open during search when servo hardware is active.
                        if eye_controller is not None:
                            # This line asks the lids to stay open while the eyes scan.
                            eye_controller.open_lids()
                            # This line moves the eye hardware through the slow search sweep.
                            eye_controller.look_at(target_lr, target_ud)
                            # This line stores the current horizontal angle for the debug overlay.
                            target_lr = eye_controller.current_lr
                            # This line stores the current vertical angle for the debug overlay.
                            target_ud = eye_controller.current_ud
                        # This line labels the status as active automatic search.
                        tracking_status = "SEARCHING"
                    # This branch returns to neutral when search is disabled or not ready to start yet.
                    else:
                        # This line turns off search mode while returning to neutral.
                        search_mode_active = False
                        # This line returns the software target to neutral after the hold time has expired.
                        target_lr = NEUTRAL_POSITION["LR"]
                        # This line returns the software target to neutral after the hold time has expired.
                        target_ud = NEUTRAL_POSITION["UD"]
                        # This line returns the eyes toward neutral when no face is visible and servo hardware is active.
                        if eye_controller is not None:
                            # This line moves the hardware back toward the neutral pose.
                            eye_controller.idle_return()
                        # This line labels the status as neutral return after face loss.
                        tracking_status = "NO FACE - RETURNING"
            # This line performs a timed blink independently of face detection when servo hardware is active.
            if eye_controller is not None and settings.auto_blink_enabled:
                # This line runs the blink scheduler.
                eye_controller.maybe_blink()
            # This line keeps any active manual blink animation moving when automatic blinking is disabled.
            elif eye_controller is not None:
                # This line advances smooth servo and lid animation at the fixed command rate.
                eye_controller.update()
            # This line prints tracking values once per configured interval.
            if now - last_debug_print >= DEBUG_PRINT_INTERVAL_SECONDS:
                # This line stores the time of the latest debug print.
                last_debug_print = now
                # This line prints live tracking values in the terminal.
                print(
                    f"face={bool(metrics['found'])} "
                    f"x={metrics['x']:.1f} y={metrics['y']:.1f} z={metrics['z']:.1f} "
                    f"x_err={metrics['x_error']:.1f} y_err={metrics['y_error']:.1f} "
                    f"target_lr={target_lr:.1f} target_ud={target_ud:.1f} "
                    f"actual_lr={(eye_controller.current_lr if eye_controller is not None else 0.0):.1f} "
                    f"actual_ud={(eye_controller.current_ud if eye_controller is not None else 0.0):.1f} "
                    f"status={tracking_status} "
                    f"manual={settings.manual_mode_enabled} "
                    f"auto_blink={settings.auto_blink_enabled} "
                    f"auto_search={settings.auto_search_enabled} "
                    f"searching={search_mode_active} "
                    f"stable_frames={stable_face_frames} "
                    f"servo_active={eye_controller is not None}",
                    flush=True,
                )
            # This line draws tracking overlays if the preview is enabled.
            if SHOW_PREVIEW:
                # This line copies the prepared frame so the preview keeps the same color correction as camtest.py.
                display_frame = frame.copy()
                # This line draws the current debug overlay only when runtime debug output is enabled.
                if settings.debug_overlay_enabled:
                    # This line draws the current debug overlay directly on the frame.
                    draw_debug(display_frame, target_face, metrics, target_lr, target_ud, eye_controller is not None, settings, tracking_status)
                # This line shows the preview window so you can see the current target.
                cv2.imshow("Eye Mechanism Camera Test", display_frame)
                # This line reads one full key event from the preview window, including arrow keys.
                key = cv2.waitKeyEx(1)
                # This line stops the loop when the user presses q or Escape.
                if key == ord("q") or key == 27:
                    # This line ends the main loop cleanly.
                    running = False
                # This line handles runtime tuning and mode toggle keys.
                else:
                    # This line applies runtime tuning or mode toggles when the key matches a tuning command.
                    tuning_message = apply_runtime_tuning_key(key, settings)
                    # This line checks whether the key was used for runtime tuning.
                    if tuning_message is not None:
                        # This line prints the tuning result so it is visible even when the overlay is hidden.
                        print(tuning_message, flush=True)
                    # This line handles keyboard servo testing only when the eye controller is available.
                    elif eye_controller is not None:
                        # This line moves the horizontal keyboard target left when the left arrow is pressed.
                        if key in KEY_LEFT_CODES:
                            # This line reduces the horizontal target by one keyboard step inside the calibrated range.
                            target_lr = clamp(target_lr - KEYBOARD_EYE_STEP_DEGREES, min(SERVO_LIMITS["LR"]), max(SERVO_LIMITS["LR"]))
                            # This line enables temporary keyboard control so camera tracking does not overwrite the test instantly.
                            manual_override_until = now + KEYBOARD_OVERRIDE_SECONDS
                            # This line starts the smooth movement toward the new keyboard target immediately.
                            eye_controller.look_at(target_lr, target_ud)
                        # This line moves the horizontal keyboard target right when the right arrow is pressed.
                        elif key in KEY_RIGHT_CODES:
                            # This line increases the horizontal target by one keyboard step inside the calibrated range.
                            target_lr = clamp(target_lr + KEYBOARD_EYE_STEP_DEGREES, min(SERVO_LIMITS["LR"]), max(SERVO_LIMITS["LR"]))
                            # This line enables temporary keyboard control so camera tracking does not overwrite the test instantly.
                            manual_override_until = now + KEYBOARD_OVERRIDE_SECONDS
                            # This line starts the smooth movement toward the new keyboard target immediately.
                            eye_controller.look_at(target_lr, target_ud)
                        # This line moves the vertical keyboard target up when the up arrow is pressed.
                        elif key in KEY_UP_CODES:
                            # This line reduces the vertical target by one keyboard step inside the calibrated range.
                            target_ud = clamp(target_ud - KEYBOARD_EYE_STEP_DEGREES, min(SERVO_LIMITS["UD"]), max(SERVO_LIMITS["UD"]))
                            # This line enables temporary keyboard control so camera tracking does not overwrite the test instantly.
                            manual_override_until = now + KEYBOARD_OVERRIDE_SECONDS
                            # This line starts the smooth movement toward the new keyboard target immediately.
                            eye_controller.look_at(target_lr, target_ud)
                        # This line moves the vertical keyboard target down when the down arrow is pressed.
                        elif key in KEY_DOWN_CODES:
                            # This line increases the vertical target by one keyboard step inside the calibrated range.
                            target_ud = clamp(target_ud + KEYBOARD_EYE_STEP_DEGREES, min(SERVO_LIMITS["UD"]), max(SERVO_LIMITS["UD"]))
                            # This line enables temporary keyboard control so camera tracking does not overwrite the test instantly.
                            manual_override_until = now + KEYBOARD_OVERRIDE_SECONDS
                            # This line starts the smooth movement toward the new keyboard target immediately.
                            eye_controller.look_at(target_lr, target_ud)
                        # This line starts a smooth blink when the b key is pressed.
                        elif key == ord("b"):
                            # This line asks the controller to run the non-blocking blink animation.
                            eye_controller.blink()
                        # This line opens the eyelids smoothly when the o key is pressed.
                        elif key == ord("o"):
                            # This line asks the controller to move eyelids to the calibrated open pose.
                            eye_controller.open_lids()
                        # This line closes the eyelids smoothly when the c key is pressed.
                        elif key == ord("c"):
                            # This line asks the controller to move eyelids to the calibrated closed pose.
                            eye_controller.close_lids()
            else:
                # This line prevents a hot loop when preview output is disabled.
                time.sleep(IDLE_LOOP_DELAY_SECONDS)
    except KeyboardInterrupt:
        # This line records a clean manual stop from the keyboard.
        running = False
    finally:
        # This line checks whether servo hardware was active before trying to move it.
        if eye_controller is not None:
            # This line moves the eye mechanism back to neutral open pose smoothly before shutdown.
            eye_controller.smooth_shutdown()
        # This line closes the camera resource.
        camera.close()
        # This line checks whether servo hardware was active before releasing it.
        if eye_controller is not None:
            # This line releases the PCA9685 resource.
            eye_controller.close()
        # This line closes any OpenCV preview windows that were opened.
        cv2.destroyAllWindows()
    # This line returns a success exit code to the shell.
    return 0


# This guard ensures the test loop only starts when the file is run directly.
if __name__ == "__main__":
    # This line exits the process using the return value from main.
    raise SystemExit(main())
