from tkinter import *
import numpy as np
import NeuralNetwork as nn
from PIL import Image, ImageOps
import io


def draw(event):
    x1, y1 = (event.x - 20), (event.y - 20)
    x2, y2 = (event.x + 20), (event.y + 20)
    c.create_oval(x1, y1, x2, y2, fill="black", outline="black")


def clear(event=None):
    c.delete("all")
    res.set(value='')


def preprocess_image():
    # Save the canvas as a PostScript file
    ps = c.postscript(colormode='mono')

    # Convert PostScript to an image
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img = img.convert("RGB")  # Convert explicitly to RGB mode
    img = img.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    img = ImageOps.invert(img)  # Invert colors so black digits on white background

    # Prepare input for neural network
    inputs = np.asarray(img).reshape(1, 784)
    inputs = inputs / 255.0 * 0.99 + 0.01
    return inputs


def recognize():
    inputs = preprocess_image()
    outputs = nnet.query(inputs)
    label = np.argmax(outputs)
    res.set(value=str(label))


def train():
    inputs = preprocess_image()
    correct_label = int(correct_digit.get())

    # Create target array with 0.01 for all values and 0.99 for the correct label
    targets = np.zeros(10) + 0.01
    targets[correct_label] = 0.99

    # Train the neural network on this single example
    nnet.train(inputs, targets)

    # Save the weights after training
    nnet.save_weights()
    print("Saved new trained data")


# Ініціалізація нейронної мережі та завантаження ваг
nnet = nn.NeuralNetwork()
nnet.load_weights("weights_input_hidden.npy", "weights_hidden_output.npy")

# Створення головного вікна
root = Tk()
root.title("HandWrittenDigitRecognizer")
root.geometry("1150x700+400+250")
root.resizable(False, False)

# Полотно для малювання
c = Canvas(root, bg="white", bd=1, highlightbackground="black")
c.place(x=50, y=50, width=500, height=500)
c.bind("<B1-Motion>", draw)
c.bind("<Button-3>", clear)

# Поле для відображення результату
res = StringVar(root, value='')
Entry(root, borderwidth=3, textvariable=res, justify="center", font=("Helvetica", 300), state="disabled").place(x=600,
                                                                                                                y=50,
                                                                                                                width=500,
                                                                                                                height=500)

# Кнопка розпізнавання
Button(root, bg="#E1E1E1", font=("BankGothic Md BT", 38), text="Recognize", command=recognize).place(x=50, y=583,
                                                                                                     width=500,
                                                                                                     height=80)

# Кнопка навчання
Button(root, bg="#E1E1E1", font=("BankGothic Md BT", 38), text="Train", command=train).place(x=600, y=583, width=500,
                                                                                             height=80)

# Список для вибору правильної цифри для навчання
correct_digit = StringVar(root)
correct_digit.set("0")  # Початкове значення
OptionMenu(root, correct_digit, *range(10)).place(x=600, y=670, width=500, height=30)

root.mainloop()
