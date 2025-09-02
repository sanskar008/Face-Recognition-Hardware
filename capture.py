import cv2
import os

person_name = input("Enter the name of the person: ")
person_dir = os.path.join('faces', person_name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Image Processing Concepts
def apply_histogram_equalization(gray_img):
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized_img = np.interp(gray_img.flatten(), range(256), cdf_normalized).reshape(gray_img.shape)
    return equalized_img.astype(np.uint8)

def apply_median_filter(gray_img, k=5):
    pad = k // 2
    padded_img = np.pad(gray_img, pad, mode='edge')
    output = np.zeros_like(gray_img)
    
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            window = padded_img[i:i+k, j:j+k]
            output[i, j] = np.median(window)
    
    return output


def gaussian_kernel(size, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def convolve(img, kernel):
    k = kernel.shape[0] // 2
    padded = np.pad(img, k, mode='edge')
    output = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+2*k+1, j:j+2*k+1]
            output[i, j] = np.sum(region * kernel)
    
    return output

def apply_canny_edge_detection(gray_img, t1=100, t2=200):
    blurred = convolve(gray_img, gaussian_kernel(5, 1))
    
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Gx = convolve(blurred, Kx)
    Gy = convolve(blurred, Ky)
    
    magnitude = np.hypot(Gx, Gy)
    angle = np.arctan2(Gy, Gx) * 180 / pi
    angle[angle < 0] += 180
    
    output = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            angle_deg = angle[i, j]
            try:
                if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                    q, r = magnitude[i, j+1], magnitude[i, j-1]
                elif (22.5 <= angle_deg < 67.5):
                    q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
                elif (67.5 <= angle_deg < 112.5):
                    q, r = magnitude[i+1, j], magnitude[i-1, j]
                else:
                    q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]
                
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    output[i, j] = magnitude[i, j]
                else:
                    output[i, j] = 0
            except:
                pass

    strong = (output >= t2)
    weak = ((output >= t1) & (output < t2))
    
    result = np.zeros_like(output)
    result[strong] = 255
    result[weak] = 75
    
    return result.astype(np.uint8)


count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_filename = os.path.join(person_dir, f'{count}.jpg')
        cv2.imwrite(face_filename, face)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Capture Face Images', frame)

    if count >= 30:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
