@echo off

REM Create folders
mkdir input
mkdir src
mkdir models
mkdir notebooks

REM Create files
copy NUL input\train.csv
copy NUL input\test.csv
copy NUL src\create_folds.py
copy NUL src\train.py
copy NUL src\inference.py
copy NUL src\models.py
copy NUL src\config.py
copy NUL src\model_dispatcher.py
copy NUL models\model_rf.bin
copy NUL models\model_et.bin
copy NUL notebooks\exploration.ipynb
copy NUL notebooks\check_data.ipynb
copy NUL README.md
copy NUL LICENSE

echo Folder structure created successfully.
