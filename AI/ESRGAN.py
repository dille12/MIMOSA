import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from train import load_upscaling_data
# ESRGAN components
def upsample_block(x_in, num_filters):
    x = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D(size=2)(x)
    return x

def residual_dense_block(x_in, num_filters):
    # Store the input for dense connections
    inputs = [x_in]
    x = x_in
    
    # Create 5 convolutional layers with dense connections
    for i in range(5):
        if i > 0:
            x = layers.Concatenate()(inputs)
        x = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        inputs.append(x)
    
    # 1x1 conv to compress feature maps
    x = layers.Concatenate()(inputs)
    x = layers.Conv2D(num_filters, kernel_size=1, padding='same')(x)
    
    # Residual connection
    return layers.Add()([x_in, x * 0.2])

def residual_in_residual_dense_block(x_in, num_filters):
    x = x_in
    for _ in range(3):
        x = residual_dense_block(x, num_filters)
    # Residual connection
    return layers.Add()([x_in, x * 0.2])

# VGG Feature Extractor for perceptual loss
def build_vgg_feature_extractor():
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    vgg.trainable = False
    
    # Extract features from specific layers for perceptual loss
    output_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in output_layers]
    
    model = Model(inputs=vgg.input, outputs=outputs)
    return model

# Generator
def build_generator(input_shape=(None, None, 1), num_filters=64, num_blocks=16, scale=2):
    # Input and initial feature extraction
    inputs = Input(shape=input_shape)
    
    # Convert grayscale to 3 channels if needed
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, kernel_size=1, padding='same')(inputs)
    else:
        x = inputs
    
    # Initial feature extraction
    x_init = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = x_init
    
    # Residual in Residual Dense Blocks
    for _ in range(num_blocks):
        x = residual_in_residual_dense_block(x, num_filters)
    
    # Feature reconstruction
    x = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = layers.Add()([x_init, x])
    
    # Upsampling blocks
    for _ in range(int(scale/2)):  # For 2x, we need 1 upsample block
        x = upsample_block(x, num_filters)
    
    # Final output layer
    if input_shape[-1] == 1:
        outputs = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    else:
        outputs = layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='generator')
    return model

# Discriminator (PatchGAN)
def build_discriminator(input_shape=(None, None, 1)):
    inputs = Input(shape=input_shape)
    
    # Convert grayscale to 3 channels if needed
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, kernel_size=1, padding='same')(inputs)
    else:
        x = inputs
    
    # Series of conv blocks with increasing filters
    filters = [64, 128, 256, 512]
    
    x = layers.Conv2D(filters[0], kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for i in range(1, len(filters)):
        x = layers.Conv2D(filters[i], kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(filters[i], kernel_size=3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output layer
    x = layers.Conv2D(1, kernel_size=3, strides=1, padding='same')(x)
    
    model = Model(inputs=inputs, outputs=x, name='discriminator')
    return model

# ESRGAN model
class ESRGAN:
    def __init__(self, input_shape=(128, 128, 1), scale=2):
        self.input_shape = input_shape
        self.scale = scale
        
        # Calculate output shape based on scale
        self.target_shape = (input_shape[0] * scale, input_shape[1] * scale, input_shape[2])
        
        # Build models
        self.generator = build_generator(input_shape, scale=scale)
        self.discriminator = build_discriminator(self.target_shape)  # Use target shape
        self.vgg = build_vgg_feature_extractor()
        
        # Set up optimizers
        self.gen_optimizer = Adam(learning_rate=1e-4)
        self.disc_optimizer = Adam(learning_rate=1e-4)
        
        # Compile discriminator
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.disc_optimizer
        )
        
    def perceptual_loss(self, real, fake):
        # Convert to RGB if grayscale
        if real.shape[-1] == 1:
            real_rgb = tf.concat([real, real, real], axis=-1)
            fake_rgb = tf.concat([fake, fake, fake], axis=-1)
        else:
            real_rgb = real
            fake_rgb = fake
        
        # Extract VGG features
        real_features = self.vgg(real_rgb)
        fake_features = self.vgg(fake_rgb)
        
        # Calculate perceptual loss (MSE between feature maps)
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += tf.reduce_mean(tf.square(real_feat - fake_feat))
        
        return loss
    
    def train_step(self, low_res_images, high_res_images):
        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate high-res images
            fake_images = self.generator(low_res_images, training=True)
            
            # Get discriminator outputs
            real_output = self.discriminator(high_res_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            # Calculate discriminator loss
            real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_output), real_output))
            fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_output), fake_output))
            disc_loss = real_loss + fake_loss
        
        # Apply gradients to discriminator
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generate high-res images
            fake_images = self.generator(low_res_images, training=True)
            
            # Get discriminator output for fake images
            fake_output = self.discriminator(fake_images, training=False)
            
            # Calculate adversarial loss
            adversarial_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output))
            
            # Calculate content loss (pixel-wise loss)
            content_loss = tf.reduce_mean(tf.square(high_res_images - fake_images))
            
            # Calculate perceptual loss
            percep_loss = self.perceptual_loss(high_res_images, fake_images)
            
            # Calculate total generator loss
            gen_loss = 0.001 * adversarial_loss + 0.1 * content_loss + 1.0 * percep_loss
        
        # Apply gradients to generator
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return {
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
            'adversarial_loss': adversarial_loss,
            'content_loss': content_loss,
            'perceptual_loss': percep_loss
        }
    
    def train(self, train_dataset, epochs=50, steps_per_epoch=None):
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Initialize metrics
            metrics = {
                'disc_loss': 0,
                'gen_loss': 0,
                'adversarial_loss': 0,
                'content_loss': 0,
                'perceptual_loss': 0
            }
            
            # Train for one epoch
            step = 0
            for low_res_batch, high_res_batch in train_dataset:
                # Perform training step
                step_metrics = self.train_step(low_res_batch, high_res_batch)
                
                # Update metrics
                for key in metrics:
                    metrics[key] += step_metrics[key]
                
                # Print progress
                step += 1
                if step % 10 == 0:
                    print(f"  Step {step}/{len(train_dataset)}: Disc Loss: {step_metrics['disc_loss']:.4f}, Gen Loss: {step_metrics['gen_loss']:.4f}")
                
                # Break if steps_per_epoch is specified
                if steps_per_epoch and step >= steps_per_epoch:
                    break
            
            # Calculate average metrics
            for key in metrics:
                metrics[key] /= step
            
            # Print epoch summary
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Discriminator Loss: {metrics['disc_loss']:.4f}")
            print(f"    Generator Loss: {metrics['gen_loss']:.4f}")
            print(f"    Adversarial Loss: {metrics['adversarial_loss']:.4f}")
            print(f"    Content Loss: {metrics['content_loss']:.4f}")
            print(f"    Perceptual Loss: {metrics['perceptual_loss']:.4f}")
    
    def save_models(self, generator_path="esrgan_generator.h5", discriminator_path="esrgan_discriminator.h5"):
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
    
    def load_generator(self, generator_path="esrgan_generator.h5"):
        self.generator = tf.keras.models.load_model(generator_path)
    
    def predict(self, low_res_images):
        return self.generator.predict(low_res_images)


# Example usage
def create_tf_dataset(inputs, targets, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=1000)
    return dataset

def train_esrgan_model(inputs, targets, input_shape=(128, 128, 1), batch_size=8, epochs=50):
    # Create TensorFlow dataset
    train_dataset = create_tf_dataset(inputs, targets, batch_size)
    
    # Initialize ESRGAN model
    esrgan = ESRGAN(input_shape=input_shape, scale=2)
    
    # Train the model
    esrgan.train(train_dataset, epochs=epochs)
    
    # Save the model
    esrgan.save_models()
    
    return esrgan

def evaluate_model(esrgan, test_inputs, test_targets):
    # Generate super-resolution images
    predicted = esrgan.predict(test_inputs)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    psnr_values = []
    for i in range(len(test_inputs)):
        psnr = tf.image.psnr(test_targets[i], predicted[i], max_val=1.0)
        psnr_values.append(psnr)
    
    avg_psnr = tf.reduce_mean(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Visualize results
    import matplotlib.pyplot as plt
    num_samples = min(5, len(test_inputs))
    
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        # Display input image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(test_inputs[i].squeeze(), cmap='gray')
        plt.title("Low Resolution")
        plt.axis('off')
        
        # Display predicted image
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(predicted[i].squeeze(), cmap='gray')
        plt.title("Super Resolution")
        plt.axis('off')
        
        # Display target image
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(test_targets[i].squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #esrgan = ESRGAN()
    img_dir = "AI/train_images/"
    images, targets = load_upscaling_data(img_dir, stride=256, max_images=500)
    train_esrgan_model(images, targets, batch_size=4)