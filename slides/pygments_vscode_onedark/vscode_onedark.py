# vscode_onedark.py
# Pygments style approximating VS Code One Dark Pro for C/C++.
from pygments.style import Style
from pygments.token import (
    Text, Comment, Keyword, Name, String, Number, Operator, Punctuation,
    Generic, Literal, Whitespace
)

# Palette (One Dark Pro-ish)
FG          = "#ABB2BF"
BG          = "#282C34"   # background
COMMENT     = "#5C6370"
KEYWORD     = "#C678DD"   # purple
TYPE        = "#E5C07B"   # yellow (types)
FUNC        = "#61AFEF"   # blue (function names)
VAR         = "#E06C75"   # red-ish (identifiers you want to stand out)
STRING      = "#98C379"   # green
NUMBER      = "#D19A66"   # orange
OP          = "#56B6C2"   # cyan (operators, e.g., &&, ::, ->)
PUNC        = FG          # punctuation same as fg
BUILTIN     = "#56B6C2"   # cyan for builtins / attrs

class VscodeOneDarkStyle(Style):
    default_style = ""
    background_color = BG
    highlight_color  = "#2C323C"  # selection

    styles = {
        # Base
        Text:                   FG,
        Whitespace:             "",
        Generic:                FG,

        # Comments
        Comment:                f"italic {COMMENT}",
        Comment.Preproc:        f"bold {KEYWORD}",

        # Keywords
        Keyword:                f"bold {KEYWORD}",
        Keyword.Type:           f"bold {TYPE}",
        Keyword.Namespace:      f"bold {KEYWORD}",
        Keyword.Constant:       f"bold {KEYWORD}",

        # Names
        Name:                   FG,
        Name.Function:          f"bold {FUNC}",       # function identifiers
        Name.Class:             f"bold {TYPE}",       # class/struct names
        Name.Namespace:         f"bold {TYPE}",
        Name.Builtin:           BUILTIN,
        Name.Builtin.Pseudo:    BUILTIN,
        Name.Decorator:         KEYWORD,
        Name.Constant:          VAR,
        Name.Variable:          VAR,
        Name.Attribute:         VAR,
        Name.Tag:               KEYWORD,
        Name.Label:             FG,

        # Literals
        String:                 STRING,
        Number:                 NUMBER,
        Literal:                FG,

        # Operators & punctuation
        Operator:               f"bold {OP}",         # &&, ->, ::, +, *
        Operator.Word:          f"bold {OP}",
        Punctuation:            PUNC,                 # (), {}, <>, commas, etc.

        # Make definitions pop
        Generic.Heading:        f"bold {FUNC}",
        Generic.Subheading:     FUNC,
    }
