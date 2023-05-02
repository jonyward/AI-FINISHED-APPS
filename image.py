import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "cascade.xml"

# Create the haar cascade
logoCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect logos in the image
logos = logoCascade.detectMultiScale(
    gray,
    scaleFactor=1.10,
    minNeighbors=3,
    minSize=(12, 12),
    maxSize=(250, 250)
)

print("Found {0} logos!".format(len(logos)))

# Draw a rectangle around the faces
for (x, y, w, h) in logos:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Logos Found", image)
cv2.waitKey(0)



