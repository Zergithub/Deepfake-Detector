from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import functional as F
import tkinter
import customtkinter
from tkinter import filedialog
import os
import cv2
import shutil
import threading
from time import sleep
from tkVideoPlayer import TkinterVideo
from PIL import Image

customtkinter.set_appearance_mode("light")
processor = AutoImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
model = AutoModelForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")
class App:
    def __init__(self):
        self.filename = ""
    
    def get_image(self,root):
        # Clear existing widgets on the root window
        for widget in root.winfo_children():
            widget.destroy()

        video_frame = tkinter.Frame(root)
        video_frame.place(relheight=0.4, relwidth=0.4, relx=0.32, rely=0.2)

        title = customtkinter.CTkLabel(master=root,
                                text="ZER's Deepfake Detector",
                                width=120,
                                height=25,
                                corner_radius=8
                                )
        title.configure(font=("font1", 20, "bold"))
        title.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)

        imageEntry = customtkinter.CTkLabel(master=root,
                                        width=50,
                                        height=50,
                                        corner_radius=10,
                                        text="")

        addImage = customtkinter.CTkButton(master=root,
                                            text= "ADD FILE",
                                            corner_radius=10,
                                            command= lambda :self.add_image_button(video_frame,imageEntry, root))
        addImage.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)

        fake_real = customtkinter.CTkLabel(master=root,
                                text="",
                                width=120,
                                height=25,
                                corner_radius=8
                                )
        fake_real.configure(font=("font1", 20, "bold"))
        fake_real.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)

        # Create a button to trigger the account creation process
        createButton = customtkinter.CTkButton(master=root,
                                            corner_radius=10,
                                            text="CHECK",
                                            command=lambda: self.add_image(root, imageEntry, fake_real))
        createButton.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)

    def add_image_button(self, video_frame, imageEntry, root):
        f_types = [('Jpg Files', '*.jpg'), ('Jpg Files', '*.png'), ('Jpg Files', '*.mp4')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        self.filename = filename
        for widget in video_frame.winfo_children():
            widget.destroy()
        
        if os.path.splitext(filename)[1] == ".mp4":

            # Create a video player widget
            videoplayer = TkinterVideo(master=video_frame, scaled=True)
            videoplayer.load(filename)
            videoplayer.pack(expand=True, fill="both")
            videoplayer.play()

            # Bind the "Ended" event to a function that handles the end of the video
            videoplayer.bind("<<Ended>>", lambda e: self.video_end(e, videoplayer))
        else:
            img = customtkinter.CTkImage(Image.open(filename), size=(video_frame.winfo_width(), 300))
            customtkinter.CTkLabel(video_frame, image=img, text="").pack(expand=True, fill="both")

        
    def extract_frames(self, video_path, output_folder, frame_interval):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video is opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create the output folder if it doesn't exist
        import os
        os.makedirs(output_folder, exist_ok=True)

        # Read and save each frame
        for frame_number in range(1):
            ret, frame = cap.read()

            if not ret:
                print(f"Error reading frame {frame_number}")
                break

            if frame_number % frame_interval == 0:
                frame_filename = f"{output_folder}/frame_{frame_number:04d}.png"
                cv2.imwrite(frame_filename, frame)


        # Release the video capture object
        cap.release()

        print(f"Frames extracted successfully to {output_folder}")

    def add_image(self,root, imageEntry, fake_real):
        # Get user input from entry fields
        image = self.filename

        # Check for missing required fields
        if image == "":
            popup = customtkinter.CTkLabel(root, text="SOME REQUIRED FIELDS ARE MISSING!")
            popup.place(relx=0.4, rely=0.9)
            root.after(3000, lambda: self.close_pop_up(popup))
            return
        if os.path.splitext(image)[1] != ".mp4":
            image_path = image
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=1).item()
            class_mapping = {0: "real", 1: "fake"}
            predicted_label = class_mapping[predicted_class]
            print(f"Predicted Label: {predicted_label}")
            fake_real.configure(text=predicted_label.upper())
        else:
                # Create a frame to contain the video player
            shutil.rmtree("./frames", ignore_errors=True)
            self.extract_frames(image, "./frames", 100)
            count = 0
            frames = os.listdir("./frames")
            for frame in frames:
                frame_path = f"./frames/{frame}"
                frame_file = Image.open(frame_path)
                inputs = processor(images=frame_file, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_class = torch.argmax(logits, dim=1).item()
                class_mapping = {0: "real", 1: "fake"}
                predicted_label = class_mapping[predicted_class]
                if predicted_label == "real":
                    count += 1
                print(predicted_label)
            if count / len(frames) >= 0.5:
                fake_real.configure(text="REAL")
            else:
                fake_real.configure(text="FAKE")

    def video_end(self,e, video):
        pass
    #     # Destroy the video player widget
    #     video.destroy()

    def loading(self,root):
        # Clear the existing widgets in the root window
        for widget in root.winfo_children():
            widget.destroy()
        til = customtkinter.CTkLabel(master=root,
                                text="ZER's Deepfake Detector",
                                width=120,
                                height=25,
                                corner_radius=8
                                )
        til.configure(font=("font1", 20, "bold"))
        til.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        img = tkinter.PhotoImage(file="logo.png")
        # Display the logo as a background label
        background_label = tkinter.Label(root, image=img)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        background_label.image = img

        title = customtkinter.CTkLabel(master=root,
                                text="...",
                                width=150,
                                height=25,
                                corner_radius=8
                                )
        title.configure(font=("font1", 100, "bold"))
        title.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)

        thread = threading.Thread(target=App.loading_wait, args=(self,title,root))
        thread.start()
        


    def loading_wait(self,title, root):
        isTrue = False
        for i in range(1, 2000):
            if isTrue and i % 100 == 0:
                title.configure(text=".. ")
                isTrue = not isTrue
                sleep(0.2)
            elif isTrue == False and i % 100 == 0:
                title.configure(text="...")
                isTrue = not isTrue
                sleep(0.2)
        self.get_image(root)

    def close_pop_up(popup):
        # Function to close and destroy pop-up messages
        popup.destroy()

def main():
    # Create the Tkinter window
    root_tk = tkinter.Tk()
    app = App()
    root_tk.geometry("500x600")
    root_tk.title("ZER's Deepfake Detector")
    root_tk.resizable(False, False)
    icon_path = "icon.ico"  # Replace with the actual path to your icon file
    root_tk.iconbitmap(icon_path)
    app.loading(root_tk)
    # app.get_image(root_tk)
    root_tk.mainloop()

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()