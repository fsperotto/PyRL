from enum import Enum

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Center, Middle, Grid, Horizontal
from textual.widgets import Header, Footer, DataTable, Tabs, Tab, ProgressBar, ContentSwitcher, Button, Label
from textual.events import MouseScrollDown
from textual.screen import ModalScreen

from rich.text import Text

ROWS = [
    ("time", "state", "actions", "rewards")
]

AGENT_ROWS = [
    ("0", "1", "2", "3")
]

class Commands(Enum):
    """docstring for Commands."""

    QUIT = -1
    GO = 0
    PAUSE = 1
    STEP = 2


class QuitScreen(ModalScreen):
    """docstring for ResetScreen."""

    def compose(self) -> ComposeResult:
        yield Center(
            Middle(
                Button("Cancel", variant="primary", id="cancel"),
                Button("Quit", variant="error", id="quit"),
            )
        )


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.action_go()
            self.app.pop_screen()
        elif event.button.id == "quit":
            self.app.connection.send(Commands.QUIT)
            self.app.exit()

class Debugger(App):
    """docstring for Debugger."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("g", "go", "Go"),
        ("p", "pause", "Pause"),
        ("s", "step", "Step"),
        ("a", "auto_scroll", "Auto Scroll"),
    ]

    def __init__(self, total_steps: int) -> None:
        super(Debugger, self).__init__()
        self.total_steps = total_steps

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Center():
            yield ProgressBar(total=self.total_steps-1, id="sim_progress")
        yield Tabs(
            Tab("Simulation", id="sim"),
            Tab("Agent", id="agent"),
            Tab("Environement", id="env"),
        )
        with ContentSwitcher(initial="sim_container", id="content_switcher"):
            yield ScrollableContainer(DataTable(id="sim_data_table"), id="sim_container")
            yield ScrollableContainer(DataTable(id="agent_data_table"), id="agent_container")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Tabs).focus()
        self.table = self.query_one("#sim_data_table")
        self.table.add_columns(*ROWS[0])

        self.agent_table = self.query_one("#agent_data_table")
        self.agent_table.add_columns(*AGENT_ROWS[0])

        for number, row in enumerate(range(1, 36+2)):
            self.agent_table.add_row(1.5, 1.5, 1.5, 1.5, label=str(number))

        self.query_one(Header).tall = True

        self.container = self.query_one("#sim_container")
        self.progress = self.query_one("#sim_progress")

        self.set_interval(1 / 60, self.receive_data)

        self.auto_scroll = True

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        self.query_one("#content_switcher").current = event.tab.id + "_container"

    def connect(self, connection):
        self.connection = connection

    def action_quit(self):
        self.action_pause()
        self.push_screen(QuitScreen())

    def action_pause(self) -> None:
        self.connection.send(Commands.PAUSE)

    def action_go(self) -> None:
        self.connection.send(Commands.GO)

    def action_step(self) -> None:
        self.connection.send(Commands.STEP)

    def action_auto_scroll(self) -> None:
        self.auto_scroll = not self.auto_scroll

    def receive_data(self) -> None:
        if self.connection.poll():
            data = self.connection.recv()
            self.table.add_row(*data["step"])
            self.progress.advance(data["progress"])

            self.agent_table.update_cell_at((data["learning"]["x"], data["learning"]["y"]), data["learning"]["value"])

        if self.auto_scroll:
            self.container.post_message(MouseScrollDown(0, 0, 0, 0, 0, None, None, None))
