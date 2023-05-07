# Simple implementation of Classic Reinforcement Learning algorithm in C 

## Compile
List all of the Allegro libs you are using on the pkg-config line

    gcc qlearning.c main.c graphics.c -o app $(pkg-config allegro-5 allegro_font-5 allegro_primitives-5 --libs --cflags)

# Run the application

    ./app