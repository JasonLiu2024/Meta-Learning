"""ANSI escape sequences"""
# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
# https://en.wikipedia.org/wiki/ANSI_escape_code
class TextColor:
    Red_text = '\033[91m'
    Red_bg = '\033[41m'
    Green_text = '\033[32m'
    Green_bg = '\033[42m'
    Yellow_text = '\033[33m'
    Yellow_bg =    '\033[43m'
    Blue_text = '\033[34m'
    Blue_bg =    '\033[44m'
    Magenta_text = '\033[35m'
    Magenta_bg =    '\033[45m'
    BrightGreen_text = '\033[92m'
    BrightGreen_bg =    '\033[102m'

    End =       '\033[0m' # necessary
    Bold =      '\033[1m'
    Underline = '\033[4m'