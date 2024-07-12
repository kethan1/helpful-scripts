import os
import sys
import argparse
from typing import Literal

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True)
parser.add_argument("--starting-frame", type=int, required=False, default=0)
parser.add_argument("--scale", type=float, required=False, default=1.0)
parser.add_argument("--separate", type=bool, required=False, default=False)
parser.add_argument("--mask", type=bool, required=False, default=True)
parser.add_argument("--hMin", type=int, required=False, default=0)
parser.add_argument("--sMin", type=int, required=False, default=0)
parser.add_argument("--vMin", type=int, required=False, default=0)
parser.add_argument("--hMax", type=int, required=False, default=179)
parser.add_argument("--sMax", type=int, required=False, default=255)
parser.add_argument("--vMax", type=int, required=False, default=255)

args = parser.parse_args()

bounds = {
    "hMin": args.hMin,
    "sMin": args.sMin,
    "vMin": args.vMin,
    "hMax": args.hMax,
    "sMax": args.sMax,
    "vMax": args.vMax,
}


def trackbar_callback(
    bounds, value: Literal["hMin", "sMin", "vMin", "hMax", "sMax", "vMax"]
):
    def callback(val):
        bounds[value] = val
        print(
            f"(hMin = {bounds['hMin']}, sMin = {bounds['sMin']}, vMin = {bounds['vMin']})"
            f"(hMax = {bounds['hMax']}, sMax = {bounds['sMax']}, vMax = {bounds['vMax']})"
        )

    return callback


# Create a window
cv2.namedWindow("image")

# create trackbars for color change
cv2.createTrackbar("HMin", "image", 0, 179, trackbar_callback(bounds, "hMin"))
cv2.createTrackbar("SMin", "image", 0, 255, trackbar_callback(bounds, "sMin"))
cv2.createTrackbar("VMin", "image", 0, 255, trackbar_callback(bounds, "vMin"))

cv2.createTrackbar("HMax", "image", 0, 179, trackbar_callback(bounds, "hMax"))
cv2.createTrackbar("SMax", "image", 0, 255, trackbar_callback(bounds, "sMax"))
cv2.createTrackbar("VMax", "image", 0, 255, trackbar_callback(bounds, "vMax"))

# Set default value for MIN HSV trackbars.
cv2.setTrackbarPos("HMin", "image", bounds["hMin"])
cv2.setTrackbarPos("SMin", "image", bounds["sMin"])
cv2.setTrackbarPos("VMin", "image", bounds["vMin"])

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos("HMax", "image", bounds["hMax"])
cv2.setTrackbarPos("SMax", "image", bounds["sMax"])
cv2.setTrackbarPos("VMax", "image", bounds["vMax"])

source_type = None

if os.path.isfile(args.source):
    # check if source is video or image
    cap = cv2.VideoCapture(args.source)

    ret, img = cap.read()

    if not ret:
        source_type = "image"
    else:
        source_type = "video"

elif os.path.isdir(args.source):
    source_type = "directory"


def mainloop(img):
    # Set minimum and max HSV values to display
    lower = np.array([bounds["hMin"], bounds["sMin"], bounds["vMin"]])
    upper = np.array([bounds["hMax"], bounds["sMax"], bounds["vMax"]])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    output = cv2.resize(output, (0, 0), fx=args.scale, fy=args.scale)
    mask = cv2.resize(mask, (0, 0), fx=args.scale, fy=args.scale)

    # Display output image
    if args.separate:
        cv2.imshow("image1", output)
    else:
        cv2.imshow("image", output)

    if args.mask:
        cv2.imshow("mask", mask)

    key = cv2.waitKey(33) & 0xFF

    return key


if source_type is None:
    print("Invalid source path.")
    sys.exit()

if source_type != "video" and args.starting_frame != 0:
    print("WARNING: Starting frame is only applicable for videos.")


if source_type == "image":
    img = cv2.imread(args.source)

    if img is None:
        print("Invalid image file.")
        sys.exit()

    while True:
        key = mainloop(img)

        if key == ord("q"):
            sys.exit()
elif source_type == "directory":
    contents = os.listdir(args.source)

    for content in contents:
        img = cv2.imread(os.path.join(args.source, content))

        if img is None:
            print(f"WARNING: Invalid image file: {content}")
            continue

        while True:
            key = mainloop(img)

            if key == ord("q"):
                sys.exit()
            elif key == ord("n"):
                break
elif source_type == "video":
    cap = cv2.VideoCapture("crp.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.starting_frame)

    while True:
        ret, img = cap.read()

        if not ret:
            break

        while True:
            key = mainloop(img)

            # Wait longer to prevent freeze for videos.
            if key == ord("n") or key == ord("w"):
                break
            elif key == ord("s"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
                break
            elif key == ord("a") or key == ord("b"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 50)
                break
            elif key == ord("d") or key == ord("f"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 50)
                break
            elif key == ord("q"):
                sys.exit()
