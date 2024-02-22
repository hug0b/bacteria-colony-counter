import tkinter as tk
from bacteria_counter_ui import BacteriaCounterUI


def main():
    root = tk.Tk()
    app = BacteriaCounterUI(root)

    # Set icon
    root.iconbitmap("./resources/icon.ico")

    root.mainloop()


if __name__ == "__main__":
    main()
