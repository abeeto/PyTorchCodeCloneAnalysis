import cv2
import multiprocessing as _mp
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_chrome_dino.utils.wrappers import make_dino
import tensorflow as tf
from time import sleep

tf.compat.v1.flags.DEFINE_integer("width", 640, "Screen width")
tf.compat.v1.flags.DEFINE_integer("height", 480, "Screen height")
tf.compat.v1.flags.DEFINE_float("threshold", 0.6, "Threshold for score")
tf.compat.v1.flags.DEFINE_float("alpha", 0.3, "Transparent level")
tf.compat.v1.flags.DEFINE_string("pre_trained_model_path", "pretrained_model.pb", "Path to pre-trained model")
FLAGS = tf.compat.v1.flags.FLAGS

HAND_GESTURES = ["Open", "Closed"]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0,128,255)

def mario(v, lock):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    done = True
    for _ in iter(int, 1):
        if done:
            env.reset()
            with lock:
                v.value = 0
        with lock:
            u = v.value
        _, _, done, _ = env.step(u)
        env.render()
        sleep(0.01)


def detect_hands(image, graph, sess):
    input_image = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = image[None, :, :, :]
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={input_image: image})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

def main():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(FLAGS.pre_trained_model_path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    graph, sess = detection_graph, sess
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FLAGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAGS.height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()
    process = mp.Process(target=mario, args=(v, lock))
    process.start()
    for _ in iter(int, 1):
        if cv2.waitKey(10) == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        count = 0
        results = {}
        for box, score, class_ in zip(boxes[:2], scores[:2], classes[:2]):
            if score > FLAGS.threshold:
                y_min = int(box[0] *FLAGS.height)
                x_min = int(box[1] * FLAGS.width)
                y_max = int(box[2] * FLAGS.height)
                x_max = int(box[3] * FLAGS.width)
                category = HAND_GESTURES[int(class_) - 1]
                results[count] = [x_min, x_max, y_min, y_max, category]
                count += 1

        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)

            if category == HAND_GESTURES[0] and x <= FLAGS.width / 3:
                action = 7  # Left jump
                text = "Jumping left"
        
            elif category == HAND_GESTURES[0] and x > 2 * FLAGS.width / 3:
                action = 2  # Right jump
                text = "Jumping right"
            elif category == HAND_GESTURES[0] and FLAGS.width / 3 < x <= 2 * FLAGS.width / 3:
                action = 5  # Jump
                text = "Jumping"
            elif category == HAND_GESTURES[1] and x <= FLAGS.width / 3:
                action = 6  # Left
                text = "Running left"
            elif category == HAND_GESTURES[1] and x > 2 * FLAGS.width / 3:
                action = 1  # Right
                text = "Running right"
            elif category == HAND_GESTURES[1] and FLAGS.width / 3 < x <= 2 * FLAGS.width / 3:
                action = 0  # Do nothing
                text = "Staying"
            else:
                action = 0
                text = "Staying"
            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (int(FLAGS.width / 3), FLAGS.height), ORANGE, -1)
        cv2.rectangle(overlay, (int(2 * FLAGS.width / 3), 0), (FLAGS.width, FLAGS.height), ORANGE, -1)
        cv2.addWeighted(overlay, FLAGS.alpha, frame, 1 - FLAGS.alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()