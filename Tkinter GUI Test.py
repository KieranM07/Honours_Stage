import tkinter as tk
from tkinter import ttk
import PalisadeModel
from tkinter import messagebox


class tkinterApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.geometry("500x400")  # Set the window size (width x height)

        self.prediction = None

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # initialize frames to an empty array
        self.frames = {}

        for F in (HomePage, TrainPage, PredictPage, ResultPage, AboutPage, DataHelpPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(HomePage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        if cont == ResultPage:
            frame.update_label()  # Ensure prediction is properly updated
        self.title(frame.page_title)
        frame.tkraise()

    def train_model(self):
        try:
            self.min10, self.max10, self.min11, self.max11, self.fire_prediction_model = PalisadeModel.train_model()
            print("Training complete")
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "Training Failed")

    def make_prediction(self):
        try:
            self.prediction = PalisadeModel.make_prediction(self.min10,self.max10,self.min11,self.max11,self.fire_prediction_model)
            print(self.prediction)
        except Exception as e:
            print(e)
            messagebox.showerror("Error", "Error: Model not trained")

    def on_closing(self):  # ensure process ends when the window is closed
        self.destroy()
        self.quit()


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
        button3 = ttk.Button(self, text="About", command=lambda: controller.show_frame(AboutPage))
        button1.grid(row=1, column=1, padx=2, pady=10, sticky="ns")
        button2.grid(row=2, column=1, padx=2, pady=10, sticky="ns")
        button3.grid(row=3, column=1, padx=2, pady=10, sticky ="ns")


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

        button_predict = ttk.Button(self, text="Make a prediction",command=lambda: controller.show_frame(PredictPage))
        button_predict.grid(row=2, column=1, padx=10, pady=10)

        button_home = ttk.Button(self, text="Return Home", command=lambda: controller.show_frame(HomePage))
        button_home.grid(row=3, column=1, padx=10, pady=10)

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

        button_train = ttk.Button(self, text="Predict", command=lambda: [self.controller.make_prediction(), controller.show_frame(ResultPage)])
        button_train.grid(row=1, column=1, padx=10, pady=10)

        button_help = ttk.Button(self, text="How to make data", command=lambda: controller.show_frame(DataHelpPage))
        button_help.grid(row=2, column=1, padx=10, pady=10)

        button_home = ttk.Button(self, text="Return Home", command=lambda: controller.show_frame(HomePage))
        button_home.grid(row=3, column=1, padx=10, pady=10)


class ResultPage(tk.Frame):
    page_title = "Prediction result"

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.label = tk.Label(self, text=f"Prediction: {self.controller.prediction}")
        self.label.grid(row=1, column=1, padx=10, pady=10)
        button_predict = ttk.Button(self, text="Make another prediction", command=lambda: controller.show_frame(PredictPage))
        button_predict.grid(row=2, column=1, padx=10, pady=10)

        button_home = ttk.Button(self, text="Return Home", command=lambda: controller.show_frame(HomePage))
        button_home.grid(row=3, column=1, padx=10, pady=10)

    def update_label(self):
        self.label.config(text=f"Prediction: {self.controller.prediction}")

class AboutPage(tk.Frame):
    page_title = "About"
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.label = tk.Label(self, text ="About")
        self.label.grid(row=1, column=1, padx=10, pady=10)

        button_home = ttk.Button(self, text="Return Home", command=lambda: controller.show_frame(HomePage))
        button_home.grid(row=3, column=1, padx=10, pady=10)

class DataHelpPage(tk.Frame):
    page_title = "Data"
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.label = tk.Label(self, text="How to collect usable data")
        self.label.grid(row=1, column=1, padx=10, pady=10)

        button_home = ttk.Button(self, text="Return Home", command=lambda: controller.show_frame(HomePage))
        button_home.grid(row=3, column=1, padx=10, pady=10)


app = tkinterApp()
app.mainloop()

