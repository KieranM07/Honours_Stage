import tkinter as tk
from tkinter import ttk
import PalisadeModel
from tkinter import messagebox
class tkinterApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.geometry("500x400")  # Set the window size (width x height)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initialize frames to an empty array
        self.frames = {}

        for F in (HomePage, TrainPage, PredictPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(HomePage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        self.title(frame.page_title)
        frame.tkraise()

    def train_model(self):
        self.min10, self.max10, self.min11, self.max11, self.fire_prediction_model = PalisadeModel.train_model()
        print("Training complete")

    def make_prediction(self):
        try:
            PalisadeModel.make_prediction(self.min10,self.max10,self.min11,self.max11,self.fire_prediction_model)
        except:
            messagebox.showerror("Error", "Error: Model not trained")


class HomePage(tk.Frame):
    page_title = "Home Page"
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Configure grid to center elements
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        label = ttk.Label(self, text="Home Page")
        label.grid(row=0, column=1, padx=10, pady=0)

        button1 = ttk.Button(self, text="Train the model", command=lambda: controller.show_frame(TrainPage))
        button2 = ttk.Button(self, text="Make a Prediction", command=lambda: controller.show_frame(PredictPage))
        button1.grid(row=1, column=1, padx=2, pady=10, sticky="ns")
        button2.grid(row=2, column=1, padx=2, pady=10, sticky="ns")


class TrainPage(tk.Frame):
    page_title = "Train Model"
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        label = ttk.Label(self, text="Model Training")
        label.grid(row=0, column=1, padx=10, pady=0)

        button_train = ttk.Button(self, text="Train", command =lambda: self.controller.train_model())
        button_train.grid(row = 1, column =1, padx = 10,pady = 10)

        button1 = ttk.Button(self, text="Make a prediction",command=lambda: controller.show_frame(PredictPage))
        button1.grid(row=2, column=1, padx=10, pady=10)

        button2 = ttk.Button(self, text="Return Home", command=lambda: controller.show_frame(HomePage))
        button2.grid(row=3, column=1, padx=10, pady=10)

class PredictPage(tk.Frame):
    page_title = "Make a prediction"
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        label = ttk.Label(self, text="Make a prediction")
        label.grid(row=0, column=1, padx=10, pady=0)

        button_train = ttk.Button(self, text="Predict", command=lambda: self.controller.make_prediction())
        button_train.grid(row=1, column=1, padx=10, pady=10)

        button2 = ttk.Button(self, text="Return Home", command=lambda: controller.show_frame(HomePage))
        button2.grid(row=2, column=1, padx=10, pady=10)


app = tkinterApp()
app.mainloop()