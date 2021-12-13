nvcc practica3/practice-effect.cu -o practica3/practice-effect `pkg-config --cflags --libs opencv`

imagenes=('720' '1080' '4k')
for i in {0..2}; 
do 
    for block in 10 20 30 40 50 60; 
    do 
        for thread in 64 128 256 512 1024;
        do 
            for iteracion in {0..9};
            do
                practica3/practice-effect practica3/img/${imagenes[$i]}.jpg practica3/${imagenes[$i]}_output.jpg $block $thread; 
            done ;
        done ;
    done ;
done
