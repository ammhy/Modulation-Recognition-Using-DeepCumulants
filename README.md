

the dataset are as follows
ID = data['ID']
SNR = data['SNR']
labels = data['labels']
features = data['features']

1. Model Overview
This model combines 1D ResNet, Transformer, and LSTM to process and classify IQ signal data. IQ signals (in-phase component I and quadrature component Q) are commonly used in wireless communication, signal processing, and similar fields. The architecture aims to extract both local and global features from the signal, while leveraging higher-order cumulant features for improved classification accuracy.

2. Model Architecture
The model consists of the following key modules:

2.1 ResNet 1D
ResNet (Residual Network) introduces skip connections (shortcut connections) to address the vanishing gradient problem in deeper networks and accelerate training. In this model, the ResNet part is used to extract temporal features from IQ signals.

Initial Convolution Layer: Accepts input of shape (batch_size, 2, seq_len), where 2 corresponds to the I and Q components, and seq_len is the sequence length. It applies a convolution layer to extract initial features.
Residual Blocks: Each residual block consists of two convolutional layers, with skip connections that add the input signal to the output of the convolution. This helps prevent gradient issues in deeper networks.
Layer Structure: The model uses four residual blocks, with the output channels increasing at each layer. The sequence length is reduced by strides, and the final output feature dimension is d_model.
2.2 Transformer Encoder
The Transformer is used to capture global dependencies in long time-series data. Through multi-head self-attention, the model can assign different attention weights to different time steps, enabling it to learn long-range dependencies in the sequence.

Transformer Layer: Includes several Transformer encoder layers, each consisting of multi-head self-attention and feed-forward networks. After ResNet feature extraction, the output is passed through the Transformer layers to model global dependencies in the sequence.
2.3 LSTM Layer
LSTM (Long Short-Term Memory) networks are used to capture long-term dependencies in sequential data. LSTMs are very effective for time-series data and improve classification accuracy by modeling time-based relationships.

LSTM Layer: After the Transformer layers, LSTM is applied to the extracted features to further model temporal dependencies. The LSTM output is a tensor of shape (batch_size, seq_len, hidden_size), and only the last time step's output is used for the final classification.
2.4 Higher-Order Cumulant Features
To better capture the statistical properties of the signal, the model calculates third-order and fourth-order cumulants from the IQ data. These higher-order statistics provide valuable non-linear features that improve the model's classification performance.
These higher-order cumulant features are concatenated with the temporal features extracted by ResNet and Transformer for final classification.

2.5 Feature Fusion and Classification
The model combines the temporal features from ResNet + Transformer with the third-order and fourth-order cumulant features. The fused features are then passed through a fully connected (FC) layer for classification, producing the final output.

Fusion: The LSTM output and the higher-order cumulant features are concatenated. Dropout is applied for regularization, and the final output is obtained through a fully connected layer.
3. Model Summary
This model leverages various deep learning techniques:

ResNet extracts local temporal features from the IQ signal.
Transformer captures global dependencies in the sequence.
LSTM further models the temporal relationships.
Higher-Order Cumulants provide additional statistical information to improve classification.
The final output is the classification result for each sample. This model is suitable for processing IQ signal data and has potential applications in wireless communication and signal processing.

4. Key Technical Highlights
Residual Connections: Effectively addresses gradient vanishing problems in deep networks.
Transformer Encoder: Models global dependencies using self-attention mechanisms.
LSTM Network: Captures long-term dependencies in sequential data.
Higher-Order Cumulant Features: Provides additional statistical information to enhance model performance.

Innovation: Incorporating Higher-Order Cumulants
One of the innovative aspects of this model is the incorporation of higher-order cumulants (third-order and fourth-order) into the feature extraction pipeline. These higher-order statistics provide valuable non-linear information about the signal that traditional first- and second-order features (like mean and variance) may fail to capture, especially in the context of complex or non-Gaussian signals.

What are Cumulants?
Cumulants are statistical quantities that provide insight into the shape and nature of a probability distribution beyond just its mean and variance. For signals like IQ data (composed of in-phase and quadrature components), higher-order cumulants can capture the nonlinear relationships between the components that may be crucial for classification tasks.

Why Use Higher-Order Cumulants in Signal Classification?
IQ signals, especially those encountered in wireless communication, can exhibit non-linear characteristics. Simply using basic signal statistics, like the mean and variance, is insufficient to model complex signal features. Higher-order cumulants, such as the third-order and fourth-order, can provide deeper insights into the interactions between the I and Q components.

Third-Order Cumulant: Represents the co-dependency between the I and Q components by calculating their mean product. This can reveal correlations or interactions that aren’t captured by simpler statistics. It helps to highlight subtle patterns in the signal that might be key for classification but are not obvious in the raw signal.

Third-order cumulant
=
𝐸
[
𝐼
⋅
𝑄
]
Third-order cumulant=E[I⋅Q]
Fourth-Order Cumulant: Focuses on even higher-order interactions. In this model, it’s calculated as the mean of the product 
𝐼
×
𝑄
×
𝐼
I×Q×I. This can detect even more complex relationships, which is especially useful when distinguishing between classes that share some statistical similarities but differ in more subtle ways.

Fourth-order cumulant
=
𝐸
[
𝐼
⋅
𝑄
⋅
𝐼
]
Fourth-order cumulant=E[I⋅Q⋅I]
By introducing these higher-order cumulants, the model is able to capture non-Gaussianity and higher-dimensional relationships in the data that traditional linear or lower-order models might miss.

How Are These Cumulants Incorporated into the Model?
Feature Fusion: After the main feature extraction (via ResNet, Transformer, and LSTM), the third- and fourth-order cumulants are computed for each sample and treated as additional features. These higher-order statistics are concatenated with the temporal features obtained from the deep learning layers (ResNet, Transformer, and LSTM).

The final feature vector for each sample contains:

Temporal features from the ResNet + Transformer + LSTM modules, which capture local and global dependencies in the sequence.
The third- and fourth-order cumulants, which provide non-linear statistical information about the signal.
This fusion of linear (temporal) and non-linear (higher-order cumulants) features allows the model to have a more comprehensive representation of the data, improving the classification performance.

Why this is Innovative:
Most deep learning models, especially in signal processing, rely primarily on raw features (e.g., the IQ components) and basic statistics. By explicitly calculating and including third- and fourth-order cumulants, this model can detect intricate relationships between signal components that might be missed by conventional methods.
Cumulants help enhance signal representations, especially when the signals exhibit complex, non-linear interactions. This gives the model the ability to capture richer signal characteristics and improve performance on classification tasks.
Real-World Impact of Using Higher-Order Cumulants:
In practice, this model would likely perform better than traditional models, especially in tasks where the signal processing is complex, such as:

Wireless communication signal classification: Where signals can be distorted by noise, interference, and channel impairments.
Non-linear signal detection: Where the relationship between the I and Q components is non-linear and higher-order statistics are essential for classification.
Signal recognition: Especially in applications like radar, communications, or speech where signal behavior can be highly dynamic and non-Gaussian.
Empirical Evidence and Benefits:
The use of higher-order cumulants can significantly improve performance by providing more discriminative power for the classifier. These statistics are sensitive to different signal conditions, allowing the model to distinguish between subtle differences in signal characteristics that would otherwise go unnoticed.


