#!/bin/zsh

pandoc $1.md -f markdown -t pdf -s -o $1.pdf