"""Shared visual tokens for the Orbit workspace."""

SPACE_QSS = r"""
QMainWindow, QWidget { background: #090f1d; color: #dce6f5; font-family: 'Segoe UI'; font-size: 10pt; }
QLabel { background: transparent; }
QMenuBar, QMenu { background: #0c1424; color: #c7d5e9; padding: 3px; }
QMenuBar::item:selected, QMenu::item:selected { background: #1a2c45; border-radius: 5px; }
QFrame#workspaceHeader { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #14233c,stop:0.6 #0e192e,stop:1 #1d2040); border-bottom: 1px solid #2d405c; }
QLabel#workspaceEyebrow { color: #80ded9; font-size: 8pt; font-weight: 700; }
QLabel#scanTitle { color: #f1f6ff; font-size: 16pt; font-weight: 700; }
QLabel#scanDetails { color: #9bafca; font-size: 9pt; }
QLabel#scanBadge { background: #182d43; color: #b2c9e7; border: 1px solid #34506c; border-radius: 7px; padding: 5px 10px; font-size: 8pt; font-weight: 700; }
QLabel#scanBadge[quality="verified"] { color: #91e8d3; background: #14352f; border-color: #286859; }
QLabel#scanBadge[quality="review"] { color: #f5d291; background: #352c1b; border-color: #756034; }
QLabel#scanBadge[quality="invalid"] { color: #ffb5bf; background: #391f2b; border-color: #7e3e52; }
QFrame#sidebar, QScrollArea#sidebarScroll { background: #0b1322; border: none; }
QFrame#sideCard { background: #111c2e; border: 1px solid #25364e; border-radius: 12px; }
QFrame#sideCard:hover { border-color: #3a506e; }
QWidget#cardContent { background: transparent; }
QToolButton#sectionToggle { background: transparent; border: none; text-align: left; color: #afc1da; font-weight: 700; font-size: 8pt; padding: 5px 2px; min-height: 20px; }
QToolButton#sectionToggle:hover { color: #83e5df; background: #17273e; }
QLabel#title { font-size: 18pt; font-weight: 700; }
QLabel#sectionTitle { color: #c1d1e8; font-weight: 700; font-size: 9pt; }
QLabel#subtitle, QLabel#welcomeBody { color: #9badc7; font-size: 9pt; }
QLabel#hint { color: #8296b4; font-size: 8.5pt; }
QFrame#statCard { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #15243a,stop:1 #101b2e); border: 1px solid #2a405c; border-radius: 12px; }
QLabel#cardValue { font-size: 17pt; font-weight: 700; color: #adf1eb; }
QLabel#cardName { color: #9eb2ce; font-size: 8.5pt; }
QPushButton, QToolButton { background: #17263c; border: 1px solid #304864; border-radius: 8px; padding: 7px 11px; color: #dfeafa; min-height: 22px; }
QPushButton:hover, QToolButton:hover { background: #213852; border-color: #65b5bd; }
QPushButton:pressed, QToolButton:pressed { background: #102136; border-color: #8bddde; }
QPushButton:focus, QToolButton:focus { border: 1px solid #86e1d8; }
QPushButton:checked, QToolButton:checked { background: #1c454e; border-color: #50c7c4; color: #d2fff9; }
QPushButton:disabled, QToolButton:disabled { color: #62748c; background: #111b2b; border-color: #233249; }
QPushButton#primary { background: #53d5ca; color: #06232c; border-color: #70e6d9; font-weight: 700; }
QPushButton#primary:hover { background: #82ede0; border-color: #b5fff1; }
QPushButton#primary:disabled { background: #173d43; color: #607e86; border-color: #2b545b; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #0a1425; border: 1px solid #304461; border-radius: 7px; padding: 5px 7px; min-height: 23px; selection-background-color: #24545a; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #67c7c5; }
QComboBox QAbstractItemView { background: #101f34; selection-background-color: #265058; color: #e0edf9; }
QCheckBox { background: transparent; spacing: 8px; min-height: 24px; }
QTabWidget::pane { border: 1px solid #24354f; border-radius: 10px; background: #0e182a; }
QTabBar::tab { background: #0c1526; color: #98acc9; padding: 11px 13px; border: none; min-width: 64px; }
QTabBar::tab:selected { background: #14253d; color: #e9fffc; }
QTabBar::tab:hover { background: #17273f; color: #c3eee9; }
QTabBar::tab:disabled { color: #4e607c; }
QTextEdit { background: #0b1628; border: 1px solid #2a3c57; border-radius: 9px; padding: 10px; font-family: 'Consolas'; }
QListWidget, QTableWidget { background: #0c1729; border: 1px solid #2a3c57; border-radius: 9px; alternate-background-color: #101e32; gridline-color: #21334e; }
QListWidget::item, QTableWidget::item { padding: 5px; }
QListWidget::item:selected, QTableWidget::item:selected { background: #244952; color: #e2fffa; }
QHeaderView::section { background: #172840; color: #b8cbe5; border: none; border-right: 1px solid #2c405d; border-bottom: 1px solid #2c405d; padding: 8px; font-weight: 600; }
QScrollBar:vertical { background: #0b1526; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #344c6c; border-radius: 5px; min-height: 28px; }
QScrollBar:horizontal { background: #0b1526; height: 10px; }
QScrollBar::handle:horizontal { background: #344c6c; border-radius: 5px; min-width: 28px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QSplitter::handle { background: #20334e; }
QSplitter::handle:hover { background: #46727f; }
QStatusBar { background: #0a1424; color: #94b4c9; padding: 4px; }
QProgressBar { background: #0a1627; border: 1px solid #2d425d; border-radius: 7px; min-height: 20px; text-align: center; color: #e2fbf6; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #245f72,stop:1 #47848a); border-radius: 5px; }
QSlider::groove:horizontal { background: #233957; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal { background: #77e1d3; border-radius: 6px; width: 12px; margin: -4px 0; }
QToolTip { color: #e2ecfa; background: #1a2c45; border: 1px solid #496584; padding: 6px; }
QFrame#welcomeStep { background: #101e33; border: 1px solid #29415c; border-radius: 12px; }
QLabel#welcomeTitle { font-size: 31pt; font-weight: 700; color: #f1f7ff; }
QLabel#stepNumber { color: #77dcd1; font-size: 16pt; font-weight: 600; }
QLabel#stepTitle { color: #e0ecfa; font-weight: 600; }
"""

def configure_plots():
    import matplotlib
    matplotlib.rcParams.update({
        # Keep default text black for white-background scientific exports.
        # Interactive canvases receive their dark colors through _style_axis.
        "axes.prop_cycle": matplotlib.cycler(color=["#76ddd2", "#ae9af4", "#78b9ee", "#f3c78a"]),
        "font.size": 10, "lines.linewidth": 1.7,
    })


def register_ui_fonts():
    """Make Windows' UI typefaces available to Qt, including offscreen previews."""
    from pathlib import Path
    from PySide6.QtGui import QFontDatabase
    for name in ("segoeui.ttf", "segoeuib.ttf", "segoeuil.ttf", "seguisb.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
