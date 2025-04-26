import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import face_recognition
from PIL import Image, ImageTk
import os
import numpy as np
from gtts import gTTS
import pygame
import time


def speak(text):
    try:
        tts = gTTS(text=text, lang='tr')
        tts.save("temp.mp3")

        pygame.mixer.init()
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.5)

        pygame.mixer.quit()
        os.remove("temp.mp3")
    except Exception as e:
        print("Sesli uyarı başarısız:", e)


def delete_face():
    faces = [os.path.splitext(f)[0] for f in os.listdir("known_faces") if f.endswith((".jpg", ".png"))]
    if not faces:
        messagebox.showinfo("Bilgi", "Hiç kayıtlı yüz bulunamadı.")
        return

    name = simpledialog.askstring("Yüz Sil", f"Silmek istediğiniz ismi girin:\n{', '.join(faces)}")
    if not name:
        return

    file_path_jpg = os.path.join("known_faces", f"{name}.jpg")
    file_path_png = os.path.join("known_faces", f"{name}.png")

    if os.path.exists(file_path_jpg):
        os.remove(file_path_jpg)
        messagebox.showinfo("Başarılı", f"{name}.jpg başarıyla silindi.")
    elif os.path.exists(file_path_png):
        os.remove(file_path_png)
        messagebox.showinfo("Başarılı", f"{name}.png başarıyla silindi.")
    else:
        messagebox.showerror("Hata", f"{name} bulunamadı!")


def add_new_face():
    name = simpledialog.askstring("Yeni Kayıt", "Yeni kişinin adını girin:")
    if not name:
        return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_img = frame[top:bottom, left:right]
            save_path = os.path.join("known_faces", f"{name}.jpg")
            cv2.imwrite(save_path, face_img)
            messagebox.showinfo("Başarılı", f"{name} başarıyla kaydedildi!")
            break

        cv2.imshow("Yüz algılanıyor... Çıkmak için ESC'ye bas", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


class FaceRecognitionApp:
    def __init__(self, root):
        self.cap = None
        self.root = root
        self.root.title("Yüz Tanıma Sistemi")
        self.root.geometry("1000x750")
        self.root.configure(bg="#0a0f2c")

        self.load_button = tk.Button(root, text="Yüzleri Yükle", command=self.load_known_faces,
                                     font=("Helvetica", 14, "bold"),
                                     bg="#1a237e", fg="white",
                                     activebackground="#3949ab", activeforeground="white",
                                     bd=0, highlightthickness=0)
        self.load_button.pack(pady=10)

        self.start_button = tk.Button(root, text="Kamerayı Başlat", command=self.start_camera,
                                      font=("Helvetica", 14, "bold"),
                                      bg="#1a237e", fg="white",
                                      activebackground="#3949ab", activeforeground="white",
                                      bd=0, highlightthickness=0)
        self.start_button.pack(pady=10)

        self.add_face_button = tk.Button(root, text="Yeni Yüz Ekle", command=add_new_face,
                                         font=("Helvetica", 14, "bold"),
                                         bg="#1a237e", fg="white",
                                         activebackground="#3949ab", activeforeground="white",
                                         bd=0, highlightthickness=0)
        self.add_face_button.pack(pady=10)

        self.delete_face_button = tk.Button(root, text="Yüz Sil", command=delete_face,
                                            font=("Helvetica", 14, "bold"),
                                            bg="#1a237e", fg="white",
                                            activebackground="#3949ab", activeforeground="white",
                                            bd=0, highlightthickness=0)
        self.delete_face_button.pack(pady=10)

        self.known_face_encodings = []
        self.known_face_names = []
        self.video_running = False
        self.unknown_saved = False
        self.greeted_names = set()
        self.prev_time = time.time()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_known_faces(self):
        directory = "known_faces"
        if not os.path.exists(directory):
            messagebox.showerror("Hata", "'known_faces' klasörü bulunamadı!")
            return

        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                path = os.path.join(directory, filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])
        messagebox.showinfo("Başarılı", "Yüzler başarıyla yüklendi!")

    def start_camera(self):
        if not self.known_face_encodings:
            messagebox.showwarning("Uyarı", "Lütfen önce yüzleri yükleyin.")
            return

        self.video_running = True
        self.cap = cv2.VideoCapture(0)

        # --- BURAYA EKLİYORUZ ---
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # -------------------------

        self.process_frame()

    def process_frame(self):
        if not self.video_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        # Frame'i küçült (çok büyük çözünürlükler FPS'i düşürür)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Küçük frame üstünde işlemler yapacağız
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Haar Cascade ile küçük resimde yüz algıla
        faces = self.face_cascade.detectMultiScale(gray_small_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Koordinatları büyüt
            x *= 2
            y *= 2
            w *= 2
            h *= 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if not self.unknown_saved:
                face_img = frame[y:y + h, x:x + w]
                save_path = os.path.join("known_faces", "face.jpg")
                cv2.imwrite(save_path, face_img)
                print("Yüz fotoğrafı 'face.jpg' olarak kaydedildi.")
                self.unknown_saved = True

        # Face recognition işlemleri
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            name = "Bilinmiyor"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            if name != "Bilinmiyor" and name not in self.greeted_names:
                speak(f"Hoş geldin {name}")
                self.greeted_names.add(name)

                # --- TANINAN KİŞİ İÇİN BİLDİRİM ---
                messagebox.showinfo("Yüz Algılandı", f"{name} algılandı!")

            if name == "Bilinmiyor" and not self.unknown_saved:
                y1 = top * 2
                x2 = right * 2
                y2 = bottom * 2
                x1 = left * 2
                face_img = frame[y1:y2, x1:x2]
                save_path = os.path.join("known_faces", "face.jpg")
                cv2.imwrite(save_path, face_img)
                self.unknown_saved = True
                speak("Tanınmayan yüz tespit edildi")

                # --- TANINMAYAN KİŞİ İÇİN BİLDİRİM ---
                messagebox.showwarning("Uyarı", "Tanınmayan kişi algılandı!")

            # Koordinatları büyüt
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # FPS yazdır
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Görüntüyü ekranda göster
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.root.after(10, self.process_frame)

    def on_close(self):
        self.video_running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
