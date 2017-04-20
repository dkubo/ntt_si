#!/bin/sh
if [ -e "$HOME/.pyenv" ]; then
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
fi
pyenv global anaconda-2.4.0
