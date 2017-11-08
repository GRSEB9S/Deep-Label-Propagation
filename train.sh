c=1
while [[ -s file.txt ]] ; # Stop when file.txt has no more lines
do
    echo "Python script called $c times"
    python train.py # Uses file.txt and removes lines from it
    c=$(($c + 1))
done
