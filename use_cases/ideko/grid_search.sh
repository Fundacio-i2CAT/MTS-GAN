CURRENT_PATH="/data/ideko_david/GSconfigs/learning_rates"
VENV_PATH="/data/ideko_david/environments/tts_gan"

source "$VENV_PATH/bin/activate"
export PYTHONPATH=$PYTHONPATH:/home/david_garcia/tts-gan/

for file_path in $CURRENT_PATH/*.yaml; do
    python main.py --config "$file_path"
done

deactivate