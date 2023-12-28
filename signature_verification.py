import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#def save_image(save_path, filename_orig, preprocessed_image):
#    save_name = f"processed_{filename_orig}"
#    save_full_path = os.path.join(save_path, save_name)
#    cv2.imwrite(save_full_path, preprocessed_image)

def save_image(save_path, filename_orig, preprocessed_image, predicted_label):
    save_name = f"processed_pred{predicted_label}_{filename_orig}"
    save_full_path = os.path.join(save_path, save_name)
    cv2.imwrite(save_full_path, preprocessed_image)

def preprocess_image(image):
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1):
        return image

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    denoised_image = cv2.medianBlur(gray_image, 3)

    return denoised_image

def process_orig_images(folder_path, save_path, desired_dim):
    orig_images = []
    orig_labels = []

    for filename_orig in sorted(os.listdir(folder_path)):
        if filename_orig.endswith(".png"):
            full_path = os.path.join(folder_path, filename_orig)

            image = cv2.imread(full_path)

            preprocessed_image = preprocess_image(image)

            #save_image(save_path, filename_orig, preprocessed_image)

            resized_image = cv2.resize(preprocessed_image, desired_dim)

            orig_images.append(resized_image)
            orig_labels.append(1)  # 1 označava ispravan potpis

    return np.array(orig_images), np.array(orig_labels)

def process_forg_images(folder_path, save_path, desired_dim):
    forg_images = []
    forg_labels = []

    for filename_forg in sorted(os.listdir(folder_path)):
        if filename_forg.endswith(".png"):
            full_path = os.path.join(folder_path, filename_forg)

            image = cv2.imread(full_path)

            preprocessed_image = preprocess_image(image)

            #save_image(save_path, filename_orig, preprocessed_image)

            resized_image = cv2.resize(preprocessed_image, desired_dim)

            forg_images.append(resized_image)
            forg_labels.append(0)  # 0 označava neispravan potpis

    return np.array(forg_images), np.array(forg_labels)

def main():
    orig_folder_path = ('C:\\Users\\Korisnik\\Desktop\\FOI\\BS\\Projekt\\archive\\signatures\\full_org')
    forg_folder_path = ('C:\\Users\\Korisnik\\Desktop\\FOI\\BS\\Projekt\\archive\\signatures\\full_forg')

    #test_orig_folder_path = ('C:\\Users\\Korisnik\\Desktop\\FOI\\BS\\Projekt\\test\\orig')
    #test_forg_folder_path = ('C:\\Users\\Korisnik\\Desktop\\FOI\\BS\\Projekt\\test\\forg')
    save_path = ('C:\\Users\\Korisnik\\Desktop\\FOI\\BS\\Projekt\\test')

    width = 424
    height = 250
    desired_dim = (width, height)

    # obrada slika originalnih potpisa
    X_train_orig, y_train_orig = process_orig_images(orig_folder_path, save_path, desired_dim)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_train_orig, y_train_orig, test_size=0.2, random_state=42)

    # obrada slika krivotvorenih potpisa
    X_train_forg, y_train_forg = process_forg_images(forg_folder_path, save_path, desired_dim)
    X_train_forg, X_test_forg, y_train_forg, y_test_forg = train_test_split(X_train_forg, y_train_forg, test_size=0.2, random_state=42)

    X_train = np.concatenate((X_train_orig, X_train_forg), axis=0)
    y_train = np.concatenate((y_train_orig, y_train_forg), axis=0)
    X_test = np.concatenate((X_test_orig, X_test_forg), axis=0)
    y_test = np.concatenate((y_test_orig, y_test_forg), axis=0)

    svm_model = SVC()
    svm_model.fit(X_train.reshape(len(X_train), -1), y_train)
    predictions = svm_model.predict(X_test.reshape(len(X_test), -1))
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy on all signatures: {accuracy}')

    far = sum(predictions[y_test == 0]) / len(predictions[y_test == 0])
    frr = 1 - (sum(predictions[y_test == 1]) / len(predictions[y_test == 1]))
    print(f'False Acceptance Rate: {far}')
    print(f'False Rejection Rate: {frr}')

    predictions = svm_model.predict(X_test.reshape(len(X_test), -1))

    for i in range(len(predictions)):
        if predictions[i] != y_test[i]:
            if y_test[i] == 1:
                save_image(save_path, f"orig_{i}.png", X_test[i], predictions[i])
            else:
                save_image(save_path, f"forg_{i}.png", X_test[i], predictions[i])

if __name__ == "__main__":
    main()