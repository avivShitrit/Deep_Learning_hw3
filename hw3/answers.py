r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=128,
        seq_len=64,
        h_dim=1024,
        n_layers=3,
        dropout=0.4,
        learn_rate=0.001,
        lr_sched_factor=0.3,
        lr_sched_patience=2,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT II"
    temperature = 0.4
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

The reason is that if we would train our model on the entire text the model will learn how to generate text's instead of words or chars, this will cause a generalization error and fix the model to specific text.
When we split the text we can make a more "creative" model that will learn the writing style instead of memorizing texts.

"""

part1_q2 = r"""
**Your answer:**


When generating text the model also uses the data stored in the hidden states,
the hidden states after training also contain data regarding connections between sequences
so the model can generate text that is longer than the sequence length due to those connections.

"""

part1_q3 = r"""
**Your answer:**


During training, the model remembers the connections between sequences and their order of appearance
which is very important when writing text, if we change the order of batches during training the model
would get confused and eventually will generate random text that has no meaning.

"""

part1_q4 = r"""
**Your answer:**

1. We use lower temperature when sampling using the model because we want the letters the model chooses to be as accurate as possible,
using low temperature allows us to use only letters the has more distinct probabilities that suppose to be more accurate.
The reason we use high temperature to train our model is that we want it to try also letters that has 
similar probabilities and "explore" each one of them to find the more accurate one.

2. High temperature creates more equally spread probabilities so the model can try different paths to find the
most accurate letter to choose, because there is no obvious accurate answer in that scenario.

3. Low temperature creates less equally spread probabilities so the model has very limited choice determining what the next letter will be, 
because low probabilities will be very close to zero and high probabilities will be even higher.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 64
    hypers['z_dim'] = 16
    hypers['x_sigma2'] = 0.0009
    hypers['learn_rate'] = 0.0001
    hypers['betas'] = (0.9, 0.999)

    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


The $ \sigma ^2 $ hyper parameter:
1. Measures how much uncertainty we have in the generation of the instance given its latent representation. 
The bigger $ /sigma $ is, the bigger the uncertainty, the less the data is “trust-worthy”.   
2. Interpreted as the strength of the regularisation term. The bigger $ /sigma $ is, the stronger the regularisation is.  
In other words, $ \sigma ^2 $ controls the weight of the reconstruction loss in the loss function.    
Low values mean that there is more weight on the reconstruction loss, thus the generative output is closer - similar to 
the input data.    
Very low values mean that the reconstruction loss is the dominant term in the function loss, and the KL divergence loss 
will have very low impact, which can result in a sparse distribution.

"""

part2_q2 = r"""
**Your answer:**


1. VAE loss can be separated into 2 distinct parts:  
The first being the Reconstruction loss, which acts as the Data term. The purpose of the Reconstruction term is to 
increase the probability that the model will reconstruct images that will be approximately close to the original data-set.
 The second being the KL divergence loss, which acts as the Regularisation term. The purpose of the Regularisation term 
is to regularise the organisation of the latent space, by making the distributions returned by the encoder close to a std 
normal distribution. 

2. When training the model we minimise the KL loss, we do so in order to achieve a minimisation between the posterior 
distribution and the prior distribution. The KL loss term encourages the encoder to distribute all encodings evenly around 
the centre of the latent space, therefore the latent space is affected directly by the KL loss. 

3. The benefit of using the KL loss is that the latent space is normally distributed. Therefore if the model distributes 
the encoding sparsely, away from the origin, it will be penalised. Thus, avoiding overfitting and ensuring that the latent 
space has good properties that enable generative process. If the KL loss wasn’t used, we might get a sparse distribution 
which from that the decoder will not be able to reconstruct $ z$  since we get a latent space that is different from the 
training-data.   


"""

part2_q3 = r"""
**Your answer:**


Plainly, upon receiving a sample of the latent space, we want our model to generate an image in the instance space.   
The training objective of the model is to maximise the probability of $ p(X) $- the evidence.
Since the evidence distribution is the distribution of the instance space, we want our model, in the training process, 
to be able to learn the very best parameters that will maximise the evidence distribution and therefore increase the 
performance of the models generative ability.

"""

part2_q4 = r"""
**Your answer:**


By definition $ \sigma $ is a positive real number, to make sure through out the model it stays that way we could use ReLU, 
but as we learned the gradient isn’t well defined around zero (smooth). Also, the std deviation values are 
very small $ 0 < \sigma << 1 $. To optimise very small numbers isn’t an easy task due to floating point arithmetic and
the poorly defined gradient, which cause numerical instability. 
When using log transform, we map the numerically unstable very small numbers in the [1, 0] interval to the [-inf, log(1)] 
interval. Calculating $ log $ and $ \exp $ is numerically stable. With this transformation we gain stability for the optimiser to work.  
On a side note, we also need the $ \log(\sigma) $ calculation for the KL divergence term. 

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=8,
        z_dim=256,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0003,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0003,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
#     hypers['discriminator_optimizer']['lr'] = 0.001
#     hypers['generator_optimizer']['lr'] = 0.001
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


The GAN containing two different models, the discriminator that should decide if an input image is real or fake
and a generator that should produce an image from a random tensor. each has its parameters but in our code,
we train them at the same time together. when we train the discriminator we want that only the discriminator
parameters will change so we use no_grad when we sample the generator at this point because the generator
samples at this stage of the training only used to feed the discriminator "fake" images but when we want to train
the generator we want its parameters to change so we calculate its gradients and update his parameters accordingly.

"""

part3_q2 = r"""
**Your answer:**


1. Because we train the generator and the discriminator together,
   if the generator loss is low it only means that the discriminator thinks that 
   the generator is producing real images but it doesn't necessarily say that the 
   discriminator is right and the images look real. the generator loss being below 
   some threshold doesn't say much and we can't get to a conclusion based on this value alone.
2. It means that the discriminator is getting ahead in the learning process and it leaves the generator behind, 
   meaning the generator keeps creating images that the discriminator easily classifies as fake so the generator's
   images don't contribute to the discriminator learning rate.

"""

part3_q3 = r"""
**Your answer:**


We can see differences in two aspects between the VAE model and the GAN model.
1. images diversity: As we can see from the results the images created by the VAE model are 
   less diverse than the GAN model, they all have the same look in the same direction with different 
   backgrounds and the GAN model creates different faces with different facial expressions.
2. image quality: the images created by the VAE is more sharp and smooth as the GAN generated more blurry images.

"""

# ==============
