#include <utility>

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

// ---------------------------------------------------------------------------

// Here, let's typedef a sample type for the siamese network. A sample is simply
// a pair of byte images.
typedef std::pair<dlib::matrix<unsigned char>,dlib::matrix<unsigned char>> image_pair;

// The following demonstrates how a user can define their own custom input layer
// to accommodate whatever data they expect. This layer's main task is to
// translate your iterable data samples into a [dlib::resizeable_]tensor. Below,
// the <interface> tag will denote things that are required to be defined for
// the dlib input layer interface. This interface is more explicitly shown in
// dlib's EXAMPLE_INPUT_LAYER code.
class input_image_pair {
public:
    // The sample expansion factor is the ratio between each sample and the
    // number of samples that will appear in the output data tensor. Since each
    // sample is a pair of images, let us define the sample expansion factor to
    // be 2.
    const static unsigned int sample_expansion_factor = 2; // <interface>
    typedef image_pair input_type;                         // <interface>

    // This function defines how we take an iterable set of inputs and convert
    // it to a data tensor.
    template <typename input_iterator>                     // <interface>
    void to_tensor(
        input_iterator ibegin,
        input_iterator iend,
        dlib::resizable_tensor& data
    ) const
    {
        // These asserts enforce different things we expect from our data.
        // While not required, this is good practice for catching silly
        // mistakes.
        DLIB_CASSERT(std::distance(ibegin, iend) > 0,"");

        // First, we extract the shape each image, and use it to define the size
        // of the data tensor.
        const long nr = ibegin->first.nr();
        const long nc = ibegin->first.nc();
        data.set_size(std::distance(ibegin, iend)*2, 1, nr, nc);

        for (auto i = ibegin; i != iend; ++i) {
            DLIB_CASSERT(i->first.nc() == nc && i->second.nc() == nc &&
                         i->first.nr() == nr && i->second.nr() == nr,"");
        }


        // To get the actual data elements of the tensor, we can call the host()
        // function. This returns a pointer to the first float element. The
        // tensor is structured in the following order: columns, rows, columns,
        // images. Therefore, the offset from image to image is
        // columns*rows*channels, channel to channel is columns*rows, etc...
        long offset = nr*nc;
        float* data_ptr = data.host();
        for (auto i = ibegin; i != iend; ++i) {
            for (long r = 0; r < nr; ++r) {
                for (long c = 0; c < nc; ++c) {
                    // Copy the data pointer while also iterating to the next
                    // element.
                    float* p = data_ptr++;
                    *p = static_cast<float>(i->first(r,c))/256.0;
                    *(p+offset) = static_cast<float>(i->second(r,c))/256.0;
                }
            }
            // In the loop above, we've already populated each image pair, so
            // here we jump to the next image pair.
            data_ptr += offset;
        }
    }
};

// Here, we provide functions for saving, loading, and being verbose about our
// custom input layer.
void serialize(const input_image_pair& item, std::ostream& out) // <interface>
{
    dlib::serialize("input_image_pair", out);
}

void deserialize(input_image_pair& item, std::istream& in)      // <interface>
{
    std::string version;
    dlib::deserialize(version, in);
    if (version != "input_image_pair") {
        throw dlib::serialization_error("Unexpected version found while deserializing input_image_pair.");
    }
}

std::ostream& operator<<(std::ostream& out, const input_image_pair& item) // <interface>
{
    out << "input_image_pair";
    return out;
}

void to_xml(const input_image_pair& item, std::ostream& out)   // <interface>
{
    out << "<input_image_pair/>";
}

// ---------------------------------------------------------------------------

// The following demonstrates how a user can define their own custom loss
// layers. This layer will be responsible for taking the network output and
// computing a loss and gradient for element in the output. Below, the
// <interface> tag will denote things that are required to be defined for the
// dlib loss layer interface. This interface is more explicitly shown in dlib's
// EXAMPLE_LOSS_LAYER_ code.
class loss_contrastive_ {
public:
    // The sample expansion factor here defines the ratio of samples to
    // labels. Since each sample pair is associated with a label, the factor is
    // set to 2.
    const static unsigned int sample_expansion_factor = 2;
    typedef unsigned char label_type;

    // Contrastive loss is defined as
    //    loss = 0.5/num_samples * sum(y*d*d + (1-y)*pow(max(margin-d,0),2))
    // where d is the euclidean distance between two samples.
    loss_contrastive_(double margin_=1.0, double thresh_=1.0)
        : margin(margin_), thresh(thresh_)
    { }

    // The label threshold here just defines a distance with which we say image
    // pairs are the same or not.
    double get_label_threshold() const
    {
        return thresh;
    }

    void set_label_threshold(double thresh_)
    {
        thresh = thresh_;
    }

    // This function defines how this layer can convert the output of a network
    // into a label. Here, we just define a sample to be the same image if the
    // distance is under a particular threhold.
    template <
        typename SUB_TYPE,
        typename label_iterator
        >
    void to_label(                           // <interface>
        const dlib::tensor& input_tensor,
        const SUB_TYPE& sub,
        label_iterator iter
    ) const
    {
        const dlib::tensor& output_tensor = sub.get_output();

        DLIB_CASSERT(output_tensor.nr() == 1 && 
                     output_tensor.nc() == 1 ,"");
        DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples(),"");

        // We iterate through each data pair and calculate distances; these are
        // then compared with a label threshold.
        const float* out_data = output_tensor.host();
        long offset = output_tensor.k();
        for (long i = 0; i < output_tensor.num_samples()/2; i += 2) {
            float d = 0;
            for (long k = 0; k < output_tensor.k(); ++k) {
                float temp = out_data[i*offset+k] - out_data[(i+1)*offset+k];
                d += temp*temp;
            }
            *(iter++) = (std::sqrt(d) < thresh) ? 1 : 0;
        }
    }

    // This function defines how this layer propagates the loss back to the rest
    // of the network. Here, we will walk through how this is specifically done.
    template <
        typename const_label_iterator,
        typename SUBNET
        >
    double compute_loss_value_and_gradient(  // <interface>
        const dlib::tensor& input_tensor,
        const_label_iterator truth,
        SUBNET& sub
    ) const
    {
        // These calls get references to the output tensor of the network and
        // the gradient to be passed back.
        const dlib::tensor& output_tensor = sub.get_output();
        dlib::tensor& grad = sub.get_gradient_input();

        // Enforce expectations of our data. For this application, we expect the
        // data to be a vector across the channels. However, if the task was for
        // semantic segmentation, the output could be an entire image instead.
        DLIB_CASSERT(output_tensor.nr() == 1 && 
                     output_tensor.nc() == 1,"");
        DLIB_CASSERT(grad.nr() == 1 && 
                     grad.nc() == 1,"");

        // Enforce interface expectations.
        DLIB_CASSERT(input_tensor.num_samples() != 0,"");
        DLIB_CASSERT(input_tensor.num_samples()%sample_expansion_factor == 0,"");
        DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples(),"");
        DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples(),"");

        // Here, we implement the contrastive loss as described above. The
        // things to note here are the pointers to the network data and it's
        // gradient. The gradient is structured similar to other tensors in the
        // library (i.e. columns, rows, channels, samples). Since, columns and
        // rows are 1 for this application, jumping from sample to sample is
        // simply the number of channels.
        const double scale = 1.0/output_tensor.num_samples();
        double loss = 0;
        const float* out_data = output_tensor.host();
        float* g = grad.host();
        for (long i = 0; i < output_tensor.num_samples()/2; i += 2) {
            const float y = *truth++;

            // Make sure the labels are 0 or 1
            DLIB_CASSERT(y == +1 || y == 0, "y: " << y);

            // x1 and x2 are populated with the network outputs of the currently
            // observed pair of samples.
            dlib::matrix<float,0,1> x1, x2;
            x1.set_size(output_tensor.k());
            x2.set_size(output_tensor.k());
            for (long k = 0; k < output_tensor.k(); ++k) {
                x1 = out_data[i*output_tensor.k()+k];
                x2 = out_data[(i+1)*output_tensor.k()+k];
            }

            float d = dlib::length(x1-x2); // euclidean distance of x1 and x2
            if (y) {
                loss += d*d;
                // NOTE: 2.0 comes from the derivative of squared(x1-x2)
                float gscale = 2.0*scale;
                for (long k = 0; k < output_tensor.k(); ++k) {
                    g[i*output_tensor.k()+k] = gscale*(x1(k)-x2(k));
                    g[(i+1)*output_tensor.k()+k] = gscale*(x2(k)-x1(k));
                }
            }
            else {
                float temp = margin-d;
                // The following handles the max behavior
                if (temp > 0) {
                    loss += temp*temp;
                    // NOTE: 1e-4 prevents 0 division
                    float gscale = -2.0*scale*temp/(d+1e-4);
                    for (long k = 0; k < output_tensor.k(); ++k) {
                        g[i*output_tensor.k()+k] = gscale*(x1(k)-x2(k));
                        g[(i+1)*output_tensor.k()+k] = gscale*(x2(k)-x1(k));
                    }
                }
                else {
                    for (long k = 0; k < output_tensor.k(); ++k) {
                        g[i*output_tensor.k()+k] = 0.0;
                        g[(i+1)*output_tensor.k()+k] = 0.0;
                    }
                }
            }
        }

        return loss*scale; // return the average loss
    }

    // As with the input layer, these functions must be defined for our custom
    // loss layer as well. Since we need to access internals of the layer, we
    // define these functions as friend functions.
    friend void serialize(const loss_contrastive_& item, std::ostream& out) // <interface>
    {
        dlib::serialize("loss_contrastive", out);
        dlib::serialize(item.margin, out);
        dlib::serialize(item.thresh, out);
    }

    friend void deserialize(loss_contrastive_& item, std::istream& in)      // <interface>
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version == "loss_contrastive") {
            dlib::deserialize(item.margin, in);
            dlib::deserialize(item.thresh, in);
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const loss_contrastive_& item) // <interface>
    {
        out << "loss_contrastive \t ("
            << "margin=" << item.margin
            << ", label_threshold=" << item.thresh
            << ")";
        return out;
    }

    friend void to_xml(const loss_contrastive_& item, std::ostream& out)    // <interface>
    {
        out << "<loss_contrastive"
            << " margin='" << item.margin << "'"
            << " label_threshold='" << item.thresh << "'/>";
    }
private:
    double margin;
    double thresh;
};


// The last implementation piece is to set an alias for our loss layer.
template <typename SUBNET>
using loss_contrastive = dlib::add_loss_layer<loss_contrastive_,SUBNET>;    // <interface>

// ---------------------------------------------------------------------------

// This function conveniently converts MNIST data into image pair data. It
// attempts to create a Siamese dataset with a fairly balanced number of
// positives and negatives.
void create_mnist_siamese_dataset(
    char* mnist_dir,
    std::vector<image_pair>& training_pairs,
    std::vector<unsigned char>& training_labels,
    std::vector<image_pair>& testing_pairs,
    std::vector<unsigned char>& testing_labels
)
{
    std::vector<dlib::matrix<unsigned char>> training_images_;
    std::vector<unsigned long> training_labels_;
    std::vector<dlib::matrix<unsigned char>> testing_images_;
    std::vector<unsigned long> testing_labels_;
    dlib::load_mnist_dataset(mnist_dir, training_images_, training_labels_, testing_images_,  testing_labels_);

    dlib::rand rnd;
    training_pairs.reserve(training_images_.size());
    training_labels.reserve(training_images_.size());
    for (unsigned long i = 0; i < training_images_.size(); ++i) {
        unsigned long j = rnd.get_random_64bit_number() % training_images_.size();
        double coin_flip = rnd.get_random_double();
        if (coin_flip >= 0.5) { // get a positive example
            while (training_labels_[i] != training_labels_[j]) {
                j = rnd.get_random_64bit_number() % training_images_.size();
            }
            training_labels.push_back(1);
        }
        else { // get a negative example
            while (training_labels_[i] == training_labels_[j]) {
                j = rnd.get_random_64bit_number() % training_images_.size();
            }
            training_labels.push_back(0);
        }
        // add image pair
        training_pairs.push_back(std::make_pair(training_images_[i], training_images_[j]));
    }

    testing_pairs.reserve(testing_images_.size());
    testing_labels.reserve(testing_images_.size());
    for (unsigned long i = 0; i < testing_images_.size(); ++i) {
        unsigned long j = rnd.get_random_64bit_number() % testing_images_.size();
        double coin_flip = rnd.get_random_double();
        if (coin_flip >= 0.5) { // get a positive example
            while (testing_labels_[i] != testing_labels_[j]) {
                j = rnd.get_random_64bit_number() % testing_images_.size();
            }
            testing_labels.push_back(1);
        }
        else { // get a negative example
            while (testing_labels_[i] == testing_labels_[j]) {
                j = rnd.get_random_64bit_number() % testing_images_.size();
            }
            testing_labels.push_back(0);
        }
        // add image pair
        testing_pairs.push_back(std::make_pair(testing_images_[i], testing_images_[j]));
    }
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) try
{
    // This example is going to run on the MNIST dataset.
    if (argc != 2)
    {
        std::cout << "This example needs the MNIST dataset to run!" << std::endl;
        std::cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << std::endl;
        std::cout << "Download the 4 files that comprise the dataset, decompress them, and" << std::endl;
        std::cout << "put them in a folder.  Then give that folder as input to this program." << std::endl;
        return 1;
    }

    std::vector<image_pair> training_pairs;
    std::vector<unsigned char> training_labels;
    std::vector<image_pair> testing_pairs;
    std::vector<unsigned char> testing_labels;
    create_mnist_siamese_dataset(argv[1], training_pairs, training_labels, testing_pairs, testing_labels);

    // We define the neural network structure here. This structure is similar to
    // the one defined in the Caffe example (in
    // mnist_siamese_train_test.prototxt). The final output that is fed in to
    // the contrastive layer is a 2-vector.
    using net_type = loss_contrastive<
                         dlib::fc<2,
                         dlib::fc<10,
                         dlib::relu<dlib::fc<500,
                         dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<50,5,5,1,1,
                         dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<20,5,5,1,1,
                         input_image_pair>>>>>>>>>>>;

    // This instantiates the defined network and we set the bias learning rate
    // multiplier to 2 to match the Caffe implementation.
    net_type net;
    dlib::layer<1>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::fc<2,...
    dlib::layer<2>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::fc<10,...
    dlib::layer<4>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::fc<500,...
    dlib::layer<7>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::con<50,...
    dlib::layer<10>(net).layer_details().set_bias_learning_rate_multiplier(2); // dlib::con<20,...

    // This pushes the network description to standard out.
    std::cout << "This network has " << net.num_layers << " layers in it." << std::endl;
    std::cout << net << std::endl;

    // We make a trainer that uses SGD.
    dlib::dnn_trainer<net_type,dlib::sgd> trainer(net, dlib::sgd(0.0,0.9));
    trainer.set_learning_rate(0.01);
    trainer.set_mini_batch_size(64);
    trainer.set_max_num_epochs(100);
    trainer.be_verbose();

    // This saves the training progress in an synchronization file every 20
    // seconds.
    trainer.set_synchronization_file("mnist_siamese_sync", std::chrono::seconds(20));

    // This sets the learning rate policy for this trainer. dlib's trainer has a
    // special policy that only shrinks the learning rate when there has been no
    // progress in a specified number of batch iterations.
    trainer.set_iterations_without_progress_threshold(2500);
    trainer.set_learning_rate_shrink_factor(0.9);

    // Train the network using the MNIST data.
    trainer.train(training_pairs, training_labels);

    // Save the network to disk. The clean call removes saved states that aren't
    // necessary for proceeding with training.
    net.clean();
    dlib::serialize("mnist_siamese_network.dat") << net;

    
    // Let us now get training and testing results for this particular label
    // threshold.
    double thresh = 1.25;

    // This shows that how the label threshold defined above in the
    // loss_contrastive layer can be modified.
    dlib::layer<0>(net).loss_details().set_label_threshold(thresh);
    std::cout << "Results for label_threshold=" << thresh << std::endl;

    std::vector<unsigned char> predicted_labels = net(training_pairs);
    int num_true_positives = 0;
    int num_false_positives = 0;
    int num_true_negatives = 0;
    int num_false_negatives = 0;
    for (std::size_t i = 0; i < training_pairs.size(); ++i) {
        if (training_labels[i] == 1) {
            if (predicted_labels[i] == 1) {
                ++num_true_positives;
            }
            else {
                ++num_false_positives;
            }
        }
        else {
            if (predicted_labels[i] == 0) {
                ++num_true_negatives;
            }
            else {
                ++num_false_negatives;
            }
        }
    }
    std::cout << "training num_true_positives: " << num_true_positives << std::endl;
    std::cout << "training num_false_positives: " << num_false_positives << std::endl;
    std::cout << "training num_true_negatives: " << num_true_negatives << std::endl;
    std::cout << "training num_false_negatives: " << num_false_negatives << std::endl;
    std::cout << "training accuracy:  "
              << (num_true_positives+num_true_negatives)/(double)(num_true_positives+
                                                                  num_false_positives+
                                                                  num_true_negatives+
                                                                  num_false_negatives) << std::endl;

    predicted_labels = net(testing_pairs);
    num_true_positives = 0;
    num_false_positives = 0;
    num_true_negatives = 0;
    num_false_negatives = 0;
    for (std::size_t i = 0; i < testing_pairs.size(); ++i) {
        if (testing_labels[i] == 1) {
            if (predicted_labels[i] == 1) {
                ++num_true_positives;
            }
            else {
                ++num_false_positives;
            }
        }
        else {
            if (predicted_labels[i] == 0) {
                ++num_true_negatives;
            }
            else {
                ++num_false_negatives;
            }
        }
    }
    std::cout << "testing num_true_positives: " << num_true_positives << std::endl;
    std::cout << "testing num_false_positives: " << num_false_positives << std::endl;
    std::cout << "testing num_true_negatives: " << num_true_negatives << std::endl;
    std::cout << "testing num_false_negatives: " << num_false_negatives << std::endl;
    std::cout << "testing accuracy:  "
              << (num_true_positives+num_true_negatives)/(double)(num_true_positives+
                                                                  num_false_positives+
                                                                  num_true_negatives+
                                                                  num_false_negatives) << std::endl;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
