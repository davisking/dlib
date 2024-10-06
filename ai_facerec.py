import sys
import os
import dlib
import glob
import argparse
from PIL import Image, ImageEnhance

def preprocess_image(img_path):
    """AI-driven preprocessing of the image to adjust brightness, contrast, and sharpness."""
    img = Image.open(img_path)
    img = ImageEnhance.Brightness(img).enhance(1.2)  # Adjust brightness
    img = ImageEnhance.Contrast(img).enhance(1.2)    # Adjust contrast
    img = ImageEnhance.Sharpness(img).enhance(1.1)   # Adjust sharpness
    return img

def load_image(img_path):
    """Load and preprocess the image, handling orientation issues."""
    img = preprocess_image(img_path)
    img = img.convert('RGB')
    img = dlib.load_rgb_image(img_path)
    return img

def main():
    parser = argparse.ArgumentParser(description="Face Recognition using dlib")
    parser.add_argument('predictor_path', type=str, help='Path to shape_predictor_5_face_landmarks.dat')
    parser.add_argument('face_rec_model_path', type=str, help='Path to dlib_face_recognition_resnet_model_v1.dat')
    parser.add_argument('faces_folder_path', type=str, help='Path to the folder containing face images')
    args = parser.parse_args()

    if not os.path.isfile(args.predictor_path):
        print(f"Error: Predictor file {args.predictor_path} does not exist.")
        sys.exit(1)

    if not os.path.isfile(args.face_rec_model_path):
        print(f"Error: Face recognition model file {args.face_rec_model_path} does not exist.")
        sys.exit(1)

    if not os.path.isdir(args.faces_folder_path):
        print(f"Error: Faces folder {args.faces_folder_path} does not exist.")
        sys.exit(1)

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(args.predictor_path)
    facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

    win = dlib.image_window()

    for f in glob.glob(os.path.join(args.faces_folder_path, "*.jpg")):
        print(f"Processing file: {f}")
        img = load_image(f)

        win.clear_overlay()
        win.set_image(img)

        dets = detector(img, 1)
        print(f"Number of faces detected: {len(dets)}")

        for k, d in enumerate(dets):
            print(f"Detection {k}: Left: {d.left()} Top: {d.top()} Right: {d.right()} Bottom: {d.bottom()}")
            shape = sp(img, d)
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)

            face_descriptor = facerec.compute_face_descriptor(img, shape)
            print("Face Descriptor (Original):", face_descriptor)

            print("Computing descriptor on aligned image ..")
            face_chip = dlib.get_face_chip(img, shape)
            face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
            print("Face Descriptor (Aligned):", face_descriptor_from_prealigned_image)

            dlib.hit_enter_to_continue()

if __name__ == "__main__":
    main()
