import re
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QTextEdit, QPushButton, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from transformers import BartForConditionalGeneration, BartTokenizer

# Загрузка предварительно обученной модели и токенизатора BART
model = BartForConditionalGeneration.from_pretrained("checkpoints_bart/checkpoint")
tokenizer = BartTokenizer.from_pretrained("checkpoints_bart/checkpoint")


def generate_summary(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(inputs, min_length=25, max_length=512, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


class GenerateThread(QThread):
    summaryGenerated = pyqtSignal(str)

    def __init__(self, input_text):
        super().__init__()
        input_text = re.sub(r'@\w+ |\[ \w+\ ]', "", input_text)
        input_text = re.sub(r'\W+', ' ', input_text)
        input_text = re.sub(r'[^a-zA-Z0-9\s]+', '', input_text)
        self.input_text = input_text

    def run(self):
        summary = generate_summary(self.input_text)
        self.summaryGenerated.emit(summary)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QGridLayout()

        self.input_text_widget = QTextEdit()
        self.layout.addWidget(QLabel("Input Text:"), 0, 0)
        self.layout.addWidget(self.input_text_widget, 1, 0)

        self.output_text_widget = QTextEdit()
        self.layout.addWidget(QLabel("Summary:"), 0, 1)
        self.layout.addWidget(self.output_text_widget, 1, 1)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar, 2, 0, 1, 2)

        self.button = QPushButton("Summarize Text")
        self.button.clicked.connect(self.summarize_text)
        self.layout.addWidget(self.button, 3, 0, 1, 2)

        self.setLayout(self.layout)

    def summarize_text(self):
        self.button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        input_text = self.input_text_widget.toPlainText()
        self.thread = GenerateThread(input_text)
        self.thread.summaryGenerated.connect(self.show_summary)
        self.thread.finished.connect(self.thread_finished)
        self.thread.start()

    def show_summary(self, summary):
        self.output_text_widget.setPlainText(summary)

    def thread_finished(self):
        self.progress_bar.setRange(0, 1)
        self.button.setEnabled(True)


app = QApplication([])
app.setStyleSheet("""
    QWidget {
        background-color: #282C34;
        color: #ABB2BF;
    }
    QLabel {font-size: 18px;}
    QTextEdit {
        background-color: #1E2128;
        border: 1px solid #1E2128;
        border-radius: 4px;
        font-size: 14px;
        padding: 8px;
    }
    QPushButton {
        background-color: #61AFEF;
        border: none;
        border-radius: 4px;
        color: #282C34;
        font-size: 16px;
        padding: 8px;
    }
    QPushButton:hover {background-color: #3F85D5;}
    QProgressBar {
        border: 1px solid #1E2128;
        border-radius: 4px;
        text-align: center;
    }

    QProgressBar::chunk {
        background-color: #61AFEF;
    }
""")
window = MainWindow()
window.resize(900, 500)
window.show()
app.exec_()
