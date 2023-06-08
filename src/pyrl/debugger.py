from enum import Enum

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Center, Middle, Grid, Horizontal, Container
from textual.widgets import Header, Footer, DataTable, Tabs, Tab, ProgressBar, ContentSwitcher, Button, Label, Static
from textual.events import MouseScrollDown
from textual.screen import ModalScreen, Screen

from rich.text import Text

ROWS = [
    ("time", "previous observation", "actions", "observation", "rewards")
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

    def __init__(self, agent, total_steps: int) -> None:
        super(Debugger, self).__init__()
        self.total_steps = total_steps
        self.agent = agent

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        return [
            Header(),
            Center(
                ProgressBar(total=self.total_steps-1, id="sim_progress")
            ),
            Tabs(
                Tab("Simulation", id="sim"),
                Tab("Agent", id="agent"),
                Tab("Environment", id="env"),
            ),
            ContentSwitcher(
                ScrollableContainer(DataTable(id="sim_data_table"), id="sim_container"),
                Container(self.agent.debugger_compose(), id="agent_container"),
                Container(Static("No data available", classes="label"), id="env_container"),
                initial="sim_container", id="content_switcher",
            ),
            Footer()
        ]

    def on_mount(self) -> None:
        self.query_one(Tabs).focus()
        self.table = self.query_one("#sim_data_table")
        self.table.add_columns(*ROWS[0])
        self.table.cursor_type = "row"

        self.container = self.query_one("#sim_container")
        self.progress = self.query_one("#sim_progress")

        self.set_interval(1 / 60, self.receive_data)

        self.query_one(Header).tall = True

        self.auto_scroll = True

        self.agent.debugger_on_mount(self)

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        self.query_one("#content_switcher").current = event.tab.id + "_container"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.agent.debugger_on_button_pressed(event)

    def connect(self, connection):
        self.connection = connection

    def action_quit(self):
        self.action_pause()
        self.push_screen(QuitScreen())

    def action_pause(self) -> None:
        self.progress.total = None
        self.connection.send(Commands.PAUSE)

    def action_go(self) -> None:
        self.progress.total = self.total_steps - 1
        self.connection.send(Commands.GO)

    def action_step(self) -> None:
        self.progress.total = None
        self.connection.send(Commands.STEP)

    def action_auto_scroll(self) -> None:
        self.auto_scroll = not self.auto_scroll

    def receive_data(self) -> None:
        if self.connection.poll():
            data = self.connection.recv()
            self.table.add_row(*data["step"])
            self.progress.advance(data["progress"])

            self.agent.debugger_update(data)

        if self.auto_scroll:
            self.container.post_message(MouseScrollDown(0, 0, 0, 0, 0, None, None, None))
