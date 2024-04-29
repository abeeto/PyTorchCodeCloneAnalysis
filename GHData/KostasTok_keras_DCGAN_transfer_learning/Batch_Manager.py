import numpy as np

class Batcher():
    '''
    Creates two minibatches, one with real images,
    and one with generated ones.
    labels, and noise for the generator.
        
    Inputs:
    data -> array (n, height, width, dim), where n is the
            number of images in data. Values are stored
            in either RGB or grayscale format.
    batch_size -> int with the total size of the batch
                  that will be used in the training process
    noise_dim -> tuple with the latent dim of the noise vector
    generator -> a compiled keras model (as in DCGAN models)
    
    Returns:
    real/fake_batch -> minibatches of real and fake images
    dis_valid_fake  -> corresponding soft labels for the discriminator
    noise -> noise vector of size double that of the batch. Half is
             used to construct the fake_batch. The vector is returned
             so that the generator can be trained with the same obs
             as the discriminator.
    gen_valid -> soft labels for the generator with the same lenght
                 as noise
    '''
    def __init__(self, data, batch_size, noise_dim, generator):
        
        # Prepare data
        self.real_inputs = (data/127.5) - 1.
        self.n = self.real_inputs.shape[0]
        self.reshuffle()
        # Initialise parameters
        if isinstance(noise_dim, int):
            self.noise_dim = (noise_dim,)
        else:
            self.noise_dim = noise_dim
        self.batch_size  = batch_size
        
        self.generator = generator
        
        # Stats on the training proces
        self.start = 0
        self.batch = 1
        self.epoch = 1
        self.times_called = 0
        
        # Create soft labels for discriminator (according to )
        self.dis_valid = np.ones(self.batch_size)*0.9
        self.dis_fake  = np.zeros(self.batch_size)
        # Create soft labels for the generator
        self.gen_valid = np.ones(self.batch_size*2)*0.9
        
    def reshuffle(self):
        # randomly reshuffle order of real images
        idx = np.random.permutation(self.n)
        self.real_inputs = self.real_inputs[idx]
        
    def next_batch(self):
        
        self.times_called += 1
        
        # Construct real batch
        end = self.start + self.batch_size
        if end <= self.n: # If not end of epoch, pass values to new array
            real_batch = self.real_inputs[self.start:end,:,:,:]
            self.batch += 1
        else: # pass remaining images after reshuffling data
            end = self.batch_size - (self.n-self.start)
            real_batch1 = self.real_inputs[self.start:,:,:,:]
            self.reshuffle()
            real_batch2 = self.real_inputs[:end,:,:,:]
            real_batch  = np.concatenate((real_batch1, real_batch2), axis=0)
            # Update stats
            self.epoch += 1
            self.batch = 1   
        self.start = end
        
        # Sample noise used to construct the fake batch
        
        noise = np.random.normal(0, 1, (self.batch_size*2,) + self.noise_dim)
        fake_batch = self.generator.predict(noise[:self.batch_size])
        
        return real_batch, fake_batch, noise
                 