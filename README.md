# MNIST CNN with Bayesian Optimization

This project trains a Convolutional Neural Network (CNN) on the MNIST dataset using Bayesian Optimization to find the best hyperparameters, such as batch size and learning rate.

Requirements
 - Python 3.x

Navigate to the project directory:
 - cd /path/to/your/project Replace /path/to/your/project with the actual path to your project directory.
 - Install dependencies: pip install -r requirements.txt

Running the Project
 - Run the following Bayesian Optimization script in the terminal: python src/bayesian_opt.py

This will:

 - Start Bayesian Optimization with different acquisition functions (EI, PI, LCB).
 - Train the CNN model for each set of hyperparameters.
 - Save results in the results/ directory as .pkl files.
 - Save trained models in the models/ directory as .pth files.
 - Generate and display plots to visualize the optimization process.