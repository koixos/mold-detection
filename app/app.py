from .state import AppState
from .pipeline.processor import Processor
from .panels.left_sidebar import LeftSidebar
from .panels.right_sidebar import RightSidebar
from .panels.portfolio import Portfolio


class MoldVisionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MoldVisionGUI")
        self.master.geometry("1600x900")

        self.processor = Processor()

        self.state = AppState()

        self.state.add_listener(self._active_changed)
        self._build_ui()


    def _build_ui(self):
        self.left = LeftSidebar(self.master, self.state)
        self.left.pack(side="left", fill="y")    

        self.portfolio = Portfolio(self.master, self.state)
        self.portfolio.pack(side="left", fill="both", expand=True)
        
        self.right = RightSidebar(self.master, self.state, self.processor)
        self.right.pack(side="right", fill="y")


    def _active_changed(self):
        self.left.refresh()
        self.portfolio.refresh()
