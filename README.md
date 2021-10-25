# CoinDetector
Coin detector with vision computer.

Detector de monedas mediante visión por computador, para poder usar el dector de monedas con yolovs5 utilizar el fichero de generate_database.py, con el cual se creará una
base de datos con todas las imagenes, divididas tanto en entrenamiento, validación y test, por último ejecutar coins_detector2.py para detectar aquellas imágenes que tengan
monedas y consiga poner las etiquetas normalizadas para yolo necesarias.

<h3>generate_dataset.py</h3>
<p>Con este script conseguiremos ampliar el tamaño de nuestro dataset y dividirlo en diferentes carpetas, siendo estas train, val y test. También se crearán 3 csv diferentes uno para cada conjunto de datos.</p>

<h3>coin_detector2.py</h3>
<p>Se tiene que realizar tanto para train, val y test el siguiente comando: </p>
<p align="center" color="blue">python .\coins_detector2.py --folder test<p>
<p>Lo que conseguiremos con esto será la generación de una carpeta label por cada carpeta train, val y test. Estas labels se encuentran normalizadas y en el formato adecuado para poder entrenar posteriormente YOLO, las labels se componen de la clase correspondiente, las coordenadas x e y normalizadas del centro de la moneda, y el tamaño y la altura también normalizadas.</p>
<br>
<p>Con esto conseguiremos una distribución como la siguiente:</p>
<br>
├───detected_coins<br>
│   ├───test<br>
│   │   ├───images<br>
│   │   └───labels<br>
│   ├───train<br>
│   │   ├───images<br>
│   │   └───labels<br>
│   └───val<br>
│       ├───images<br>
│       └───labels
