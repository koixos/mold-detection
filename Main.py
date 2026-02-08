import tkinter as tk
from app.app import MoldVisionApp

if __name__ == '__main__':
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 1.0)

    MoldVisionApp(root)
    
    root.mainloop()
