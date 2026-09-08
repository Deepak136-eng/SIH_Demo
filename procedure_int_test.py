import cv2
import numpy as np
import time
import mediapipe as mp
import pyttsx3
import threading
import queue

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# ASTRA
# AUTONOMOUS SCIENTIFIC TASK RECOGNITION & ASSISTANCE SYSTEM
#
# FEATURES
# ----------------------------------------------------------
# HAND DETECTION
# OBJECT DETECTION
# PROCEDURE RECOGNITION
# PROCEDURE VIOLATION MONITORING
# EXPERIMENT ACTIVITY LOG
# EXPERIMENT PERFORMANCE TRACKING
# OFFLINE VOICE ASSISTANT
# CONTINUOUS PROCEDURE GUIDANCE
# ERROR DETECTION AND RECOVERY GUIDANCE
# ==========================================================


# ==========================================================
# PHASE 12
# ADVANCED OFFLINE VOICE ASSISTANT
# ==========================================================

voice_queue = queue.Queue()

voice_enabled = True

last_voice_message = ""
last_voice_time = 0

VOICE_REPEAT_COOLDOWN = 3.0


def voice_worker():

    try:

        engine = pyttsx3.init()

        engine.setProperty("rate", 165)

        engine.setProperty("volume", 1.0)

    except Exception as e:

        print(f"VOICE ENGINE INITIALIZATION ERROR: {e}")

        return

    while True:

        message = voice_queue.get()

        if message is None:

            voice_queue.task_done()

            break

        try:

            print(f"ASTRA VOICE: {message}")

            engine.say(message)

            engine.runAndWait()

        except Exception as e:

            print(f"VOICE ERROR: {e}")

        voice_queue.task_done()


# ==========================================================
# START VOICE THREAD
# ==========================================================

voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)

voice_thread.start()


# ==========================================================
# SPEAK FUNCTION
# ==========================================================

def speak(message, force=False):

    global voice_enabled
    global last_voice_message
    global last_voice_time

    if not voice_enabled:

        return

    if message is None:

        return

    if message == "":

        return

    current_voice_time = time.time()

    # Prevent repeated messages
    if not force:

        if (
            message == last_voice_message
            and
            current_voice_time - last_voice_time
            < VOICE_REPEAT_COOLDOWN
        ):

            return

    last_voice_message = message

    last_voice_time = current_voice_time

    voice_queue.put(message)


# ==========================================================
# CLEAR PENDING VOICE MESSAGES
# ==========================================================

def clear_voice_queue():

    try:

        while not voice_queue.empty():

            voice_queue.get_nowait()

            voice_queue.task_done()

    except queue.Empty:

        pass


# ==========================================================
# EXPERIMENT ACTIVITY LOG
# ==========================================================

experiment_log = []

MAX_LOG_ENTRIES = 6


# ==========================================================
# EXPERIMENT PERFORMANCE VARIABLES
# ==========================================================

experiment_start_time = None

experiment_end_time = None

experiment_completed = False

experiment_duration = 0


# ==========================================================
# PROCEDURE VIOLATIONS
# ==========================================================

procedure_violation_count = 0


# ==========================================================
# CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Unable to access camera")

    exit()


# ==========================================================
# MEDIAPIPE HAND LANDMARKER
# ==========================================================

MODEL_PATH = "models/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.4,

    min_hand_presence_confidence=0.4,

    min_tracking_confidence=0.4
)


hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)


start_time = time.time()


# ==========================================================
# OBJECT DETECTION SETTINGS
# ==========================================================

MIN_OBJECT_AREA = 700


# ==========================================================
# BLACK OBJECT
# SAMPLE A
# ==========================================================

lower_black = np.array([0, 0, 0])

upper_black = np.array([180, 255, 70])


# ==========================================================
# PINK OBJECT
# EXPERIMENT VESSEL
# ==========================================================

lower_pink = np.array([140, 80, 80])

upper_pink = np.array([179, 255, 255])


# ==========================================================
# RED OBJECT
# ANALYSIS MODULE
# ==========================================================

lower_red_1 = np.array([0, 100, 80])

upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 100, 80])

upper_red_2 = np.array([180, 255, 255])


# ==========================================================
# ACTIVITY SETTINGS
# ==========================================================

INTERACTION_THRESHOLD = 100

INTERACTION_HOLD_TIME = 0.5

TRANSPORT_DISTANCE_REQUIRED = 70


# ==========================================================
# PROCEDURE SEQUENCE
# ==========================================================

expected_sequence = [

    "PICK_SAMPLE",

    "TRANSPORT_SAMPLE",

    "PLACE_SAMPLE_IN_VESSEL",

    "OPERATE_ANALYSIS_MODULE"

]


# ==========================================================
# PROCEDURE VARIABLES
# ==========================================================

current_step = 0

current_activity = "IDLE / OBSERVING"

activity_confidence = 0

procedure_status = "WAITING TO START"

procedure_color = (0, 255, 255)

assistance_message = "Pick SAMPLE-A to begin."

assistance_color = (0, 255, 255)


# ==========================================================
# STEP EVENT VARIABLES
# ==========================================================

sample_pick_start_time = None

module_interaction_start_time = None

sample_initial_position = None

sample_transport_distance = 0

sample_inside_vessel_previous = False


# ==========================================================
# STEP NOTIFICATION
# ==========================================================

last_completed_step = None

step_notification = ""

notification_start_time = 0

NOTIFICATION_DURATION = 2.5


# ==========================================================
# PROCEDURE ERROR DETECTION
# ==========================================================

procedure_error = ""

error_start_time = 0

ERROR_DURATION = 2.5

error_cooldown = 0

ERROR_COOLDOWN_DURATION = 3.0

procedure_monitor_status = "NORMAL"

procedure_monitor_color = (0, 255, 0)

last_error_message = "No procedure violations detected."


# ==========================================================
# GUIDANCE SYSTEM
# ==========================================================

last_guidance_time = 0

GUIDANCE_INTERVAL = 10.0

last_guidance_step = -1


# ==========================================================
# LOG EVENT
# ==========================================================

def log_event(event_type, message):

    global experiment_log

    timestamp = time.strftime("%H:%M:%S")

    log_entry = {

        "time": timestamp,

        "type": event_type,

        "message": message

    }

    experiment_log.append(log_entry)

    if len(experiment_log) > MAX_LOG_ENTRIES:

        experiment_log.pop(0)

    print(f"[{timestamp}] {event_type}: {message}")


# ==========================================================
# DETECT OBJECT
# ==========================================================

def detect_object(mask, min_area):

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    if not contours:

        return None, None

    largest_contour = max(

        contours,

        key=cv2.contourArea

    )

    area = cv2.contourArea(

        largest_contour

    )

    if area < min_area:

        return None, None

    x, y, w, h = cv2.boundingRect(

        largest_contour

    )

    center = (

        x + w // 2,

        y + h // 2

    )

    return (

        x,

        y,

        w,

        h

    ), center


# ==========================================================
# DRAW OBJECT
# ==========================================================

def draw_object(

    frame,

    box,

    center,

    label,

    color

):

    if box is None:

        return

    x, y, w, h = box

    cv2.rectangle(

        frame,

        (x, y),

        (x + w, y + h),

        color,

        3

    )

    cv2.circle(

        frame,

        center,

        6,

        color,

        -1

    )

    cv2.putText(

        frame,

        label,

        (

            x,

            max(25, y - 10)

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        color,

        2

    )


# ==========================================================
# CALCULATE DISTANCE
# ==========================================================

def calculate_distance(point1, point2):

    return np.sqrt(

        (point1[0] - point2[0]) ** 2

        +

        (point1[1] - point2[1]) ** 2

    )


# ==========================================================
# DISTANCE FROM HAND TO OBJECT
# ==========================================================

def distance_to_box(point, box):

    if box is None:

        return float("inf")

    px, py = point

    x, y, w, h = box

    closest_x = max(

        x,

        min(

            px,

            x + w

        )

    )

    closest_y = max(

        y,

        min(

            py,

            y + h

        )

    )

    return calculate_distance(

        (

            px,

            py

        ),

        (

            closest_x,

            closest_y

        )

    )


# ==========================================================
# SAMPLE INSIDE VESSEL
# ==========================================================

def is_sample_inside_vessel(

    sample_center,

    vessel_box

):

    if sample_center is None:

        return False

    if vessel_box is None:

        return False

    x, y, w, h = vessel_box

    sample_x, sample_y = sample_center

    margin = 15

    return (

        sample_x > x + margin

        and

        sample_x < x + w - margin

        and

        sample_y > y + margin

        and

        sample_y < y + h - margin

    )


# ==========================================================
# GET ASSISTANCE
# ==========================================================

def get_assistance(step):

    if step >= len(expected_sequence):

        return (

            "Experiment completed successfully!",

            (0, 255, 0)

        )

    messages = {

        0:
            "Pick SAMPLE-A to begin.",

        1:
            "Move SAMPLE-A away from its original position.",

        2:
            "Place SAMPLE-A inside the experiment vessel.",

        3:
            "Operate the red analysis module."

    }

    return (

        messages.get(

            step,

            "Follow the procedure."

        ),

        (0, 255, 255)

    )


# ==========================================================
# VOICE GUIDANCE
# ==========================================================

def get_voice_instruction(step):

    messages = {

        0:

            (
                "Step one. Pick up Sample A to begin "
                "the experiment."
            ),

        1:

            (
                "Step one completed successfully. "
                "Now transport Sample A away from its "
                "original position."
            ),

        2:

            (
                "Step two completed successfully. "
                "Now place Sample A inside the "
                "experiment vessel."
            ),

        3:

            (
                "Step three completed successfully. "
                "Now operate the red analysis module "
                "to finish the experiment."
            ),

        4:

            (
                "All procedure steps completed. "
                "Experiment completed successfully."
            )

    }

    return messages.get(

        step,

        "Please follow the experiment procedure."

    )


# ==========================================================
# CONTINUOUS VOICE GUIDANCE
# ==========================================================

def provide_periodic_guidance(current_time):

    global last_guidance_time
    global last_guidance_step

    if not voice_enabled:

        return

    if experiment_completed:

        return

    if procedure_error != "":

        return

    if (

        current_time - last_guidance_time

        >=

        GUIDANCE_INTERVAL

    ):

        if current_step < len(expected_sequence):

            instruction = get_voice_instruction(
                current_step
            )

            speak(instruction)

            last_guidance_time = current_time

            last_guidance_step = current_step


# ==========================================================
# REPORT PROCEDURE ERROR
# ==========================================================

def report_procedure_error(

    error_message,

    recovery_message,

    current_time

):

    global procedure_error
    global error_start_time
    global error_cooldown
    global procedure_monitor_status
    global procedure_monitor_color
    global last_error_message
    global assistance_message
    global assistance_color
    global procedure_violation_count

    if current_time < error_cooldown:

        return

    procedure_error = error_message

    error_start_time = current_time

    error_cooldown = (

        current_time

        +

        ERROR_COOLDOWN_DURATION

    )

    procedure_monitor_status = "VIOLATION DETECTED"

    procedure_monitor_color = (

        0,

        0,

        255

    )

    last_error_message = error_message

    assistance_message = recovery_message

    assistance_color = (

        0,

        165,

        255

    )

    procedure_violation_count += 1

    log_event(

        "VIOLATION",

        error_message

    )

    # ======================================================
    # CLEAR OLD VOICE INSTRUCTIONS
    # ======================================================

    clear_voice_queue()

    # ======================================================
    # VOICE ERROR AND RECOVERY
    # ======================================================

    speak(

        f"Warning. {error_message}. "
        f"{recovery_message}",

        force=True

    )

    print(

        f"PROCEDURE ERROR: {error_message}"

    )


# ==========================================================
# COMPLETE STEP
# ==========================================================

def complete_step(

    activity,

    confidence

):

    global current_step
    global current_activity
    global activity_confidence
    global procedure_status
    global procedure_color
    global assistance_message
    global assistance_color
    global last_completed_step
    global step_notification
    global notification_start_time
    global procedure_monitor_status
    global procedure_monitor_color

    global experiment_start_time
    global experiment_end_time
    global experiment_duration
    global experiment_completed

    global last_guidance_time


    if current_step >= len(expected_sequence):

        return


    completed_step_index = current_step

    completed_step_number = current_step + 1


    # ======================================================
    # START EXPERIMENT TIMER
    # ======================================================

    if (

        completed_step_index == 0

        and

        experiment_start_time is None

    ):

        experiment_start_time = time.time()

        print("EXPERIMENT TIMER STARTED")

        log_event(

            "SYSTEM",

            "Experiment timer started."

        )


    # ======================================================
    # UPDATE STEP
    # ======================================================

    current_activity = activity

    activity_confidence = confidence

    last_completed_step = completed_step_number

    current_step += 1


    procedure_status = (

        f"STEP {completed_step_number} COMPLETED"

    )

    procedure_color = (

        0,

        255,

        0

    )


    procedure_monitor_status = "NORMAL"

    procedure_monitor_color = (

        0,

        255,

        0

    )


    assistance_message, assistance_color = get_assistance(

        current_step

    )


    step_notification = (

        f"STEP {completed_step_number} COMPLETE!"

    )

    notification_start_time = time.time()


    log_event(

        "STEP COMPLETE",

        f"Step {completed_step_number}: {activity}"

    )


    print(

        f"STEP {completed_step_number} COMPLETED: {activity}"

    )


    # ======================================================
    # VOICE STEP ANNOUNCEMENT
    # ======================================================

    clear_voice_queue()

    if current_step < len(expected_sequence):

        speak(

            get_voice_instruction(

                current_step

            ),

            force=True

        )

        last_guidance_time = time.time()


    # ======================================================
    # EXPERIMENT COMPLETION
    # ======================================================

    if (

        completed_step_number == len(expected_sequence)

        and

        experiment_start_time is not None

    ):

        experiment_end_time = time.time()

        experiment_duration = (

            experiment_end_time

            -

            experiment_start_time

        )

        experiment_completed = True


        print(

            f"EXPERIMENT COMPLETED IN "

            f"{experiment_duration:.2f} SECONDS"

        )


        log_event(

            "SUCCESS",

            f"Experiment completed in "
            f"{experiment_duration:.2f} seconds."

        )


        clear_voice_queue()

        speak(

            f"Congratulations. The experiment has been "
            f"completed successfully in "
            f"{experiment_duration:.1f} seconds. "
            f"All procedure steps were completed.",

            force=True

        )


# ==========================================================
# RESET PROCEDURE
# ==========================================================

def reset_procedure():

    global current_step
    global current_activity
    global activity_confidence
    global sample_pick_start_time
    global module_interaction_start_time
    global sample_initial_position
    global sample_transport_distance
    global sample_inside_vessel_previous
    global procedure_status
    global procedure_color
    global assistance_message
    global assistance_color
    global last_completed_step
    global step_notification
    global procedure_error
    global error_start_time
    global error_cooldown
    global procedure_monitor_status
    global procedure_monitor_color
    global last_error_message
    global procedure_violation_count

    global experiment_start_time
    global experiment_end_time
    global experiment_completed
    global experiment_duration

    global last_guidance_time


    current_step = 0

    current_activity = "IDLE / OBSERVING"

    activity_confidence = 0


    sample_pick_start_time = None

    module_interaction_start_time = None

    sample_initial_position = None

    sample_transport_distance = 0

    sample_inside_vessel_previous = False


    last_completed_step = None

    step_notification = ""


    procedure_error = ""

    error_start_time = 0

    error_cooldown = 0


    procedure_monitor_status = "NORMAL"

    procedure_monitor_color = (

        0,

        255,

        0

    )


    last_error_message = (

        "No procedure violations detected."

    )


    procedure_status = "PROCEDURE RESET"

    procedure_color = (

        0,

        255,

        255

    )


    procedure_violation_count = 0


    experiment_start_time = None

    experiment_end_time = None

    experiment_completed = False

    experiment_duration = 0


    last_guidance_time = time.time()


    assistance_message, assistance_color = get_assistance(

        current_step

    )


    experiment_log.clear()


    log_event(

        "SYSTEM",

        "Procedure reset. Ready to begin."

    )


    clear_voice_queue()

    speak(

        "Procedure reset successfully. "
        "ASTRA is ready for a new experiment. "
        "Step one. Pick Sample A to begin.",

        force=True

    )


    print("ASTRA PROCEDURE RESET")


# ==========================================================
# INITIAL SYSTEM LOG
# ==========================================================

log_event(

    "SYSTEM",

    "ASTRA initialized. Waiting for experiment."

)


# ==========================================================
# STARTUP VOICE
# ==========================================================

speak(

    "ASTRA initialized successfully. "
    "Offline voice assistant is active. "
    "The experiment procedure contains four steps. "
    "Step one. Pick Sample A to begin the experiment.",

    force=True

)


# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Unable to read camera frame")

        speak(

            "Camera error detected. "
            "Unable to read the camera feed.",

            force=True

        )

        break


    # ======================================================
    # MIRROR CAMERA
    # ======================================================

    frame = cv2.flip(

        frame,

        1

    )


    height, width, _ = frame.shape

    current_time = time.time()


    # ======================================================
    # PERIODIC VOICE GUIDANCE
    # ======================================================

    provide_periodic_guidance(

        current_time

    )


    # ======================================================
    # RESTORE NORMAL STATUS AFTER ERROR
    # ======================================================

    if (

        procedure_error != ""

        and

        current_time - error_start_time

        >=

        ERROR_DURATION

    ):

        procedure_error = ""

        procedure_monitor_status = "NORMAL"

        procedure_monitor_color = (

            0,

            255,

            0

        )

        assistance_message, assistance_color = get_assistance(

            current_step

        )


    # ======================================================
    # HAND DETECTION
    # ======================================================

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame

    )


    timestamp_ms = int(

        (

            current_time

            -

            start_time

        )

        *

        1000

    )


    hand_result = hand_landmarker.detect_for_video(

        mp_image,

        timestamp_ms

    )


    hand_detected = False

    hand_center = None

    interaction_point = None


    # ======================================================
    # HAND LANDMARKS
    # ======================================================

    if hand_result.hand_landmarks:

        hand_detected = True

        landmarks = hand_result.hand_landmarks[0]

        landmark_x = []

        landmark_y = []


        for landmark in landmarks:

            x = int(

                landmark.x * width

            )

            y = int(

                landmark.y * height

            )


            landmark_x.append(x)

            landmark_y.append(y)


            cv2.circle(

                frame,

                (

                    x,

                    y

                ),

                4,

                (

                    255,

                    0,

                    255

                ),

                -1

            )


        connections = [

            (0, 1), (1, 2), (2, 3), (3, 4),

            (0, 5), (5, 6), (6, 7), (7, 8),

            (5, 9), (9, 10), (10, 11), (11, 12),

            (9, 13), (13, 14), (14, 15), (15, 16),

            (13, 17), (17, 18), (18, 19), (19, 20),

            (0, 17)

        ]


        for start, end in connections:

            start_point = (

                int(

                    landmarks[start].x * width

                ),

                int(

                    landmarks[start].y * height

                )

            )


            end_point = (

                int(

                    landmarks[end].x * width

                ),

                int(

                    landmarks[end].y * height

                )

            )


            cv2.line(

                frame,

                start_point,

                end_point,

                (

                    255,

                    0,

                    255

                ),

                2

            )


        hand_center = (

            int(

                np.mean(

                    landmark_x

                )

            ),

            int(

                np.mean(

                    landmark_y

                )

            )

        )


        interaction_point = (

            int(

                landmarks[8].x * width

            ),

            int(

                landmarks[8].y * height

            )

        )


        cv2.circle(

            frame,

            hand_center,

            8,

            (

                0,

                255,

                255

            ),

            -1

        )


        cv2.circle(

            frame,

            interaction_point,

            10,

            (

                0,

                165,

                255

            ),

            2

        )


    # ======================================================
    # OBJECT DETECTION
    # ======================================================

    hsv = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2HSV

    )


    black_mask = cv2.inRange(

        hsv,

        lower_black,

        upper_black

    )


    pink_mask = cv2.inRange(

        hsv,

        lower_pink,

        upper_pink

    )


    red_mask_1 = cv2.inRange(

        hsv,

        lower_red_1,

        upper_red_1

    )


    red_mask_2 = cv2.inRange(

        hsv,

        lower_red_2,

        upper_red_2

    )


    red_mask = cv2.bitwise_or(

        red_mask_1,

        red_mask_2

    )


    # ======================================================
    # MORPHOLOGICAL FILTERING
    # ======================================================

    kernel = np.ones(

        (

            5,

            5

        ),

        np.uint8

    )


    black_mask = cv2.morphologyEx(

        black_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    black_mask = cv2.morphologyEx(

        black_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    pink_mask = cv2.morphologyEx(

        pink_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    pink_mask = cv2.morphologyEx(

        pink_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    red_mask = cv2.morphologyEx(

        red_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    red_mask = cv2.morphologyEx(

        red_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    # ======================================================
    # DETECT OBJECTS
    # ======================================================

    black_box, black_center = detect_object(

        black_mask,

        MIN_OBJECT_AREA

    )


    vessel_box, vessel_center = detect_object(

        pink_mask,

        MIN_OBJECT_AREA

    )


    red_box, red_center = detect_object(

        red_mask,

        MIN_OBJECT_AREA

    )


    # ======================================================
    # DRAW OBJECTS
    # ======================================================

    draw_object(

        frame,

        black_box,

        black_center,

        "SAMPLE-A",

        (

            0,

            255,

            0

        )

    )


    draw_object(

        frame,

        vessel_box,

        vessel_center,

        "EXPERIMENT VESSEL",

        (

            255,

            0,

            255

        )

    )


    draw_object(

        frame,

        red_box,

        red_center,

        "ANALYSIS MODULE",

        (

            0,

            0,

            255

        )

    )


    # ======================================================
    # HAND OBJECT INTERACTION
    # ======================================================

    interaction_object = None

    closest_distance = float("inf")

    detection_point = interaction_point


    if detection_point is None:

        detection_point = hand_center


    if hand_detected and detection_point is not None:

        objects = [

            (

                "SAMPLE-A",

                black_box,

                black_center,

                (

                    0,

                    255,

                    0

                )

            ),

            (

                "EXPERIMENT VESSEL",

                vessel_box,

                vessel_center,

                (

                    255,

                    0,

                    255

                )

            ),

            (

                "ANALYSIS MODULE",

                red_box,

                red_center,

                (

                    0,

                    0,

                    255

                )

            )

        ]


        closest_object = None

        closest_center = None

        closest_color = None


        for (

            object_name,

            object_box,

            object_center,

            object_color

        ) in objects:

            if object_box is None:

                continue


            distance = distance_to_box(

                detection_point,

                object_box

            )


            if distance < closest_distance:

                closest_distance = distance

                closest_object = object_name

                closest_center = object_center

                closest_color = object_color


        if (

            closest_object is not None

            and

            closest_distance < INTERACTION_THRESHOLD

        ):

            interaction_object = closest_object


            if closest_center is not None:

                cv2.line(

                    frame,

                    detection_point,

                    closest_center,

                    closest_color,

                    3

                )


                cv2.putText(

                    frame,

                    f"INTERACTING: {interaction_object}",

                    (

                        20,

                        height - 25

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    closest_color,

                    2

                )


    # ======================================================
    # SAMPLE INSIDE VESSEL
    # ======================================================

    sample_inside_vessel = is_sample_inside_vessel(

        black_center,

        vessel_box

    )


    # ======================================================
    # PROCEDURE VIOLATION DETECTION
    # ======================================================

    if current_step == 0:

        if interaction_object == "EXPERIMENT VESSEL":

            report_procedure_error(

                "Invalid action. The vessel was accessed before picking the sample.",

                "Recovery instruction. Please pick Sample A first.",

                current_time

            )


        elif interaction_object == "ANALYSIS MODULE":

            report_procedure_error(

                "Wrong sequence. The analysis module was operated too early.",

                "Recovery instruction. Pick Sample A to begin.",

                current_time

            )


    elif current_step == 1:

        if interaction_object == "ANALYSIS MODULE":

            report_procedure_error(

                "Wrong sequence. The analysis module requires completed sample preparation.",

                "Recovery instruction. Transport Sample A first.",

                current_time

            )


        elif sample_inside_vessel:

            report_procedure_error(

                "Procedure violation. The sample was placed before transport completion.",

                "Recovery instruction. Complete sample transport first.",

                current_time

            )


    elif current_step == 2:

        if interaction_object == "ANALYSIS MODULE":

            report_procedure_error(

                "Wrong sequence. The analysis module cannot operate yet.",

                "Recovery instruction. Place Sample A inside the vessel.",

                current_time

            )


    elif current_step == 3:

        if interaction_object == "SAMPLE-A":

            report_procedure_error(

                "Invalid action. Sample preparation has already been completed.",

                "Recovery instruction. Operate the analysis module.",

                current_time

            )


        elif interaction_object == "EXPERIMENT VESSEL":

            report_procedure_error(

                "Invalid action. Vessel preparation has already been completed.",

                "Recovery instruction. Operate the analysis module.",

                current_time

            )


    # ======================================================
    # STEP 1
    # PICK SAMPLE
    # ======================================================

    if current_step == 0:

        current_activity = "WAITING FOR PICK"

        activity_confidence = 0


        if interaction_object == "SAMPLE-A":

            if sample_pick_start_time is None:

                sample_pick_start_time = current_time


            elif (

                current_time

                -

                sample_pick_start_time

                >=

                INTERACTION_HOLD_TIME

            ):

                if black_center is not None:

                    sample_initial_position = black_center


                complete_step(

                    "PICK_SAMPLE",

                    90

                )


                sample_pick_start_time = None


        else:

            sample_pick_start_time = None


    # ======================================================
    # STEP 2
    # TRANSPORT SAMPLE
    # ======================================================

    elif current_step == 1:

        current_activity = "TRANSPORT_SAMPLE"

        activity_confidence = 70


        if (

            sample_initial_position is not None

            and

            black_center is not None

        ):

            sample_transport_distance = calculate_distance(

                black_center,

                sample_initial_position

            )


            cv2.putText(

                frame,

                f"TRANSPORT: {int(sample_transport_distance)} px",

                (

                    20,

                    height - 55

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (

                    0,

                    255,

                    255

                ),

                2

            )


            if (

                sample_transport_distance

                >=

                TRANSPORT_DISTANCE_REQUIRED

            ):

                complete_step(

                    "TRANSPORT_SAMPLE",

                    88

                )


    # ======================================================
    # STEP 3
    # PLACE SAMPLE IN VESSEL
    # ======================================================

    elif current_step == 2:

        current_activity = "PLACE_SAMPLE_IN_VESSEL"

        activity_confidence = 75


        if sample_inside_vessel:

            complete_step(

                "PLACE_SAMPLE_IN_VESSEL",

                96

            )


    # ======================================================
    # STEP 4
    # OPERATE ANALYSIS MODULE
    # ======================================================

    elif current_step == 3:

        current_activity = "WAITING FOR MODULE"

        activity_confidence = 0


        if interaction_object == "ANALYSIS MODULE":

            if module_interaction_start_time is None:

                module_interaction_start_time = current_time


            elif (

                current_time

                -

                module_interaction_start_time

                >=

                INTERACTION_HOLD_TIME

            ):

                complete_step(

                    "OPERATE_ANALYSIS_MODULE",

                    92

                )


                module_interaction_start_time = None


        else:

            module_interaction_start_time = None


    # ======================================================
    # EXPERIMENT COMPLETE
    # ======================================================

    else:

        current_activity = "EXPERIMENT COMPLETED"

        activity_confidence = 100


    # ======================================================
    # ACTIVITY COLORS
    # ======================================================

    activity_colors = {

        "PICK_SAMPLE":

            (

                0,

                180,

                0

            ),

        "TRANSPORT_SAMPLE":

            (

                255,

                165,

                0

            ),

        "PLACE_SAMPLE_IN_VESSEL":

            (

                0,

                180,

                0

            ),

        "OPERATE_ANALYSIS_MODULE":

            (

                0,

                0,

                255

            ),

        "EXPERIMENT COMPLETED":

            (

                0,

                180,

                0

            )

    }


    activity_color = activity_colors.get(

        current_activity,

        (

            80,

            80,

            80

        )

    )


    # ======================================================
    # CAMERA DISPLAY
    # ======================================================

    camera_display = frame.copy()


    # ======================================================
    # CAMERA HEADER
    # ======================================================

    cv2.rectangle(

        camera_display,

        (

            0,

            0

        ),

        (

            width,

            48

        ),

        (

            30,

            30,

            30

        ),

        -1

    )


    cv2.putText(

        camera_display,

        "ASTRA | LIVE EXPERIMENT FEED",

        (

            15,

            31

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (

            255,

            255,

            255

        ),

        2

    )


    # ======================================================
    # LIVE INDICATOR
    # ======================================================

    cv2.circle(

        camera_display,

        (

            width - 65,

            23

        ),

        6,

        (

            0,

            255,

            0

        ),

        -1

    )


    cv2.putText(

        camera_display,

        "LIVE",

        (

            width - 50,

            28

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (

            0,

            255,

            0

        ),

        1

    )


    # ======================================================
    # DASHBOARD
    # ======================================================

    dashboard_width = 520

    dashboard_height = 820


    dashboard = np.zeros(

        (

            dashboard_height,

            dashboard_width,

            3

        ),

        dtype=np.uint8

    )


    dashboard[:] = (

        245,

        245,

        245

    )


    # ======================================================
    # HELPER FUNCTIONS
    # ======================================================

    def draw_card(image, x1, y1, x2, y2):

        cv2.rectangle(

            image,

            (

                x1,

                y1

            ),

            (

                x2,

                y2

            ),

            (

                255,

                255,

                255

            ),

            -1

        )


        cv2.rectangle(

            image,

            (

                x1,

                y1

            ),

            (

                x2,

                y2

            ),

            (

                210,

                210,

                210

            ),

            1

        )


    def draw_section_title(image, title, y):

        cv2.putText(

            image,

            title,

            (

                25,

                y

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (

                70,

                70,

                70

            ),

            1

        )


    # ======================================================
    # DASHBOARD HEADER
    # ======================================================

    cv2.rectangle(

        dashboard,

        (

            0,

            0

        ),

        (

            dashboard_width,

            80

        ),

        (

            35,

            35,

            35

        ),

        -1

    )


    cv2.putText(

        dashboard,

        "ASTRA SYSTEM",

        (

            25,

            34

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (

            255,

            255,

            255

        ),

        2

    )


    cv2.putText(

        dashboard,

        "Autonomous Scientific Task Recognition & Assistance",

        (

            25,

            60

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (

            190,

            190,

            190

        ),

        1

    )


    # ======================================================
    # SYSTEM STATUS CARD
    # ======================================================

    draw_card(

        dashboard,

        15,

        95,

        505,

        145

    )


    if current_step >= len(expected_sequence):

        system_status_text = "EXPERIMENT COMPLETE"

    else:

        system_status_text = "SYSTEM ACTIVE"


    cv2.circle(

        dashboard,

        (

            38,

            120

        ),

        8,

        (

            0,

            200,

            0

        ),

        -1

    )


    cv2.putText(

        dashboard,

        system_status_text,

        (

            58,

            126

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.58,

        (

            40,

            40,

            40

        ),

        2

    )


    # ======================================================
    # VIOLATION COUNT
    # ======================================================

    violation_color = (

        0,

        0,

        255

    ) if procedure_violation_count > 0 else (

        0,

        150,

        0

    )


    cv2.putText(

        dashboard,

        f"VIOLATIONS: {procedure_violation_count}",

        (

            330,

            126

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.46,

        violation_color,

        1

    )


    # ======================================================
    # CURRENT PROCEDURE CARD
    # ======================================================

    draw_card(

        dashboard,

        15,

        160,

        505,

        285

    )


    draw_section_title(

        dashboard,

        "CURRENT PROCEDURE",

        185

    )


    if current_step < len(expected_sequence):

        step_display = (

            f"STEP {current_step + 1} / "

            f"{len(expected_sequence)}"

        )

    else:

        step_display = "ALL STEPS COMPLETED"


    cv2.putText(

        dashboard,

        step_display,

        (

            25,

            220

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (

            30,

            30,

            30

        ),

        2

    )


    activity_display_names = {

        "WAITING FOR PICK":

            "Waiting to Pick Sample",

        "PICK_SAMPLE":

            "Pick Sample",

        "TRANSPORT_SAMPLE":

            "Transport Sample",

        "PLACE_SAMPLE_IN_VESSEL":

            "Place Sample in Vessel",

        "WAITING FOR MODULE":

            "Waiting for Module",

        "OPERATE_ANALYSIS_MODULE":

            "Operate Analysis Module",

        "EXPERIMENT COMPLETED":

            "Experiment Completed",

        "IDLE / OBSERVING":

            "Observing"

    }


    clean_activity = activity_display_names.get(

        current_activity,

        current_activity

    )


    cv2.putText(

        dashboard,

        clean_activity,

        (

            25,

            250

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        activity_color,

        1

    )


    # ======================================================
    # PROGRESS BAR
    # ======================================================

    progress = current_step / len(expected_sequence)

    bar_x = 25

    bar_y = 262

    bar_width = 460

    bar_height = 14


    cv2.rectangle(

        dashboard,

        (

            bar_x,

            bar_y

        ),

        (

            bar_x + bar_width,

            bar_y + bar_height

        ),

        (

            220,

            220,

            220

        ),

        -1

    )


    filled_width = int(

        bar_width * progress

    )


    if filled_width > 0:

        cv2.rectangle(

            dashboard,

            (

                bar_x,

                bar_y

            ),

            (

                bar_x + filled_width,

                bar_y + bar_height

            ),

            (

                0,

                180,

                0

            ),

            -1

        )


    # ======================================================
    # NEXT ACTION CARD
    # ======================================================

    draw_card(

        dashboard,

        15,

        300,

        505,

        370

    )


    draw_section_title(

        dashboard,

        "NEXT ACTION",

        325

    )


    if current_step < len(expected_sequence):

        next_action_names = {

            "PICK_SAMPLE":

                "Pick SAMPLE-A",

            "TRANSPORT_SAMPLE":

                "Move SAMPLE-A away from starting position",

            "PLACE_SAMPLE_IN_VESSEL":

                "Place SAMPLE-A inside vessel",

            "OPERATE_ANALYSIS_MODULE":

                "Operate analysis module"

        }


        next_action = next_action_names.get(

            expected_sequence[current_step],

            "Follow procedure"

        )

    else:

        next_action = (

            "Experiment Successfully Completed"

        )


    cv2.putText(

        dashboard,

        next_action[:60],

        (

            25,

            355

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        (

            40,

            40,

            40

        ),

        1

    )


    # ======================================================
    # LIVE DETECTION CARD
    # ======================================================

    draw_card(

        dashboard,

        15,

        385,

        505,

        490

    )


    draw_section_title(

        dashboard,

        "LIVE DETECTION",

        410

    )


    status_items = [

        (

            "HAND",

            hand_detected

        ),

        (

            "SAMPLE-A",

            black_box is not None

        ),

        (

            "VESSEL",

            vessel_box is not None

        ),

        (

            "MODULE",

            red_box is not None

        )

    ]


    positions = [

        (

            35,

            440

        ),

        (

            270,

            440

        ),

        (

            35,

            470

        ),

        (

            270,

            470

        )

    ]


    for i, (

        label,

        detected

    ) in enumerate(status_items):

        item_x, item_y = positions[i]


        if detected:

            status_color = (

                0,

                200,

                0

            )

            status_text = "DETECTED"

        else:

            status_color = (

                0,

                0,

                255

            )

            status_text = "NOT DETECTED"


        cv2.circle(

            dashboard,

            (

                item_x,

                item_y

            ),

            7,

            status_color,

            -1

        )


        cv2.putText(

            dashboard,

            label,

            (

                item_x + 16,

                item_y + 5

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.46,

            (

                50,

                50,

                50

            ),

            1

        )


        cv2.putText(

            dashboard,

            status_text,

            (

                item_x + 85,

                item_y + 5

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.38,

            status_color,

            1

        )


    # ======================================================
    # PROCEDURE MONITORING CARD
    # ======================================================

    draw_card(

        dashboard,

        15,

        505,

        505,

        575

    )


    draw_section_title(

        dashboard,

        "PROCEDURE MONITORING",

        530

    )


    cv2.circle(

        dashboard,

        (

            38,

            552

        ),

        8,

        procedure_monitor_color,

        -1

    )


    cv2.putText(

        dashboard,

        procedure_monitor_status,

        (

            58,

            558

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.52,

        procedure_monitor_color,

        1

    )


    # ======================================================
    # ASTRA ASSISTANCE CARD
    # ======================================================

    draw_card(

        dashboard,

        15,

        590,

        505,

        645

    )


    cv2.putText(

        dashboard,

        "ASTRA ASSISTANCE",

        (

            25,

            614

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (

            80,

            80,

            80

        ),

        1

    )


    cv2.putText(

        dashboard,

        assistance_message[:65],

        (

            25,

            637

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.40,

        assistance_color,

        1

    )


    # ======================================================
    # VOICE STATUS
    # ======================================================

    voice_status = (

        "VOICE: ON"

        if voice_enabled

        else

        "VOICE: OFF"

    )


    voice_color = (

        0,

        180,

        0

    ) if voice_enabled else (

        0,

        0,

        255

    )


    # ======================================================
    # EXPERIMENT ACTIVITY LOG
    # ======================================================

    draw_card(

        dashboard,

        15,

        660,

        505,

        780

    )


    draw_section_title(

        dashboard,

        "EXPERIMENT ACTIVITY LOG",

        685

    )


    display_logs = experiment_log[-MAX_LOG_ENTRIES:]

    log_y = 710


    for entry in reversed(display_logs):

        event_type = entry["type"]

        timestamp = entry["time"]

        message = entry["message"]


        if event_type == "VIOLATION":

            log_color = (

                0,

                0,

                255

            )

            prefix = "[!]"

        elif event_type == "SUCCESS":

            log_color = (

                0,

                180,

                0

            )

            prefix = "[OK]"

        elif event_type == "STEP COMPLETE":

            log_color = (

                0,

                150,

                0

            )

            prefix = "[+]"

        else:

            log_color = (

                80,

                80,

                80

            )

            prefix = "[*]"


        log_text = (

            f"{timestamp} {prefix} {message}"

        )


        cv2.putText(

            dashboard,

            log_text[:72],

            (

                25,

                log_y

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.35,

            log_color,

            1

        )


        log_y += 16


        if log_y > 765:

            break


    # ======================================================
    # FOOTER
    # ======================================================

    cv2.putText(

        dashboard,

        "[R] RESET",

        (

            20,

            810

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.38,

        (

            100,

            100,

            100

        ),

        1

    )


    cv2.putText(

        dashboard,

        voice_status,

        (

            200,

            810

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.38,

        voice_color,

        1

    )


    cv2.putText(

        dashboard,

        "[Q] QUIT",

        (

            420,

            810

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.38,

        (

            100,

            100,

            100

        ),

        1

    )


    # ======================================================
    # STEP COMPLETION NOTIFICATION
    # ======================================================

    if (

        step_notification != ""

        and

        current_time - notification_start_time

        <

        NOTIFICATION_DURATION

    ):

        text_size = cv2.getTextSize(

            step_notification,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            2

        )[0]


        box_width = text_size[0] + 40

        box_height = 50


        box_x = (

            width // 2

            -

            box_width // 2

        )


        box_y = 60


        cv2.rectangle(

            camera_display,

            (

                box_x,

                box_y

            ),

            (

                box_x + box_width,

                box_y + box_height

            ),

            (

                0,

                100,

                0

            ),

            -1

        )


        cv2.rectangle(

            camera_display,

            (

                box_x,

                box_y

            ),

            (

                box_x + box_width,

                box_y + box_height

            ),

            (

                0,

                255,

                0

            ),

            2

        )


        cv2.putText(

            camera_display,

            step_notification,

            (

                box_x + 20,

                box_y + 33

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (

                255,

                255,

                255

            ),

            2

        )


    # ======================================================
    # PROCEDURE ERROR NOTIFICATION
    # ======================================================

    if (

        procedure_error != ""

        and

        current_time - error_start_time

        <

        ERROR_DURATION

    ):

        error_text = "PROCEDURE VIOLATION"


        text_size = cv2.getTextSize(

            error_text,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            2

        )[0]


        box_width = max(

            text_size[0] + 40,

            450

        )


        box_height = 85


        box_x = (

            width // 2

            -

            box_width // 2

        )


        box_y = 120


        cv2.rectangle(

            camera_display,

            (

                box_x,

                box_y

            ),

            (

                box_x + box_width,

                box_y + box_height

            ),

            (

                0,

                0,

                120

            ),

            -1

        )


        cv2.rectangle(

            camera_display,

            (

                box_x,

                box_y

            ),

            (

                box_x + box_width,

                box_y + box_height

            ),

            (

                0,

                0,

                255

            ),

            2

        )


        cv2.putText(

            camera_display,

            error_text,

            (

                box_x + 20,

                box_y + 30

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (

                255,

                255,

                255

            ),

            2

        )


        cv2.putText(

            camera_display,

            procedure_error[:60],

            (

                box_x + 20,

                box_y + 60

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.40,

            (

                255,

                255,

                255

            ),

            1

        )


    # ======================================================
    # DISPLAY WINDOWS
    # ======================================================

    cv2.imshow(

        "ASTRA - Live Experiment Feed",

        camera_display

    )


    cv2.imshow(

        "ASTRA - Procedure Dashboard",

        dashboard

    )


    # ======================================================
    # KEYBOARD CONTROLS
    # ======================================================

    key = cv2.waitKey(1) & 0xFF


    # ======================================================
    # QUIT
    # ======================================================

    if key == ord("q"):

        clear_voice_queue()

        speak(

            "ASTRA system shutting down. "
            "Thank you for using ASTRA.",

            force=True

        )

        time.sleep(2)

        break


    # ======================================================
    # RESET
    # ======================================================

    elif key == ord("r"):

        reset_procedure()


    # ======================================================
    # VOICE ON / OFF
    # ======================================================

    elif key == ord("v"):

        voice_enabled = not voice_enabled


        if voice_enabled:

            print("ASTRA VOICE ENABLED")

            speak(

                "Voice assistant enabled. "
                "ASTRA voice guidance is active.",

                force=True

            )


            log_event(

                "SYSTEM",

                "Voice assistant enabled."

            )


        else:

            print("ASTRA VOICE DISABLED")


            log_event(

                "SYSTEM",

                "Voice assistant disabled."

            )


# ==========================================================
# CLEANUP
# ==========================================================

try:

    hand_landmarker.close()

except Exception:

    pass


cap.release()

cv2.destroyAllWindows()


# ==========================================================
# STOP VOICE THREAD
# ==========================================================

voice_queue.put(None)