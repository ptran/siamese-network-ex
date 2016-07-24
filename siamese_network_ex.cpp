#include <utility>

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

// ---------------------------------------------------------------------------

typedef std::pair<dlib::matrix<unsigned char>,dlib::matrix<unsigned char> > image_pair;

class input_image_pair {
public:
    const static unsigned int sample_expansion_factor = 2;
    // We define the input type as a pointer to a matrix. This allows
    // construction of pairwise examples without copying input data.
    typedef image_pair input_type;

    template <typename input_iterator>
    void to_tensor(
        input_iterator ibegin,
        input_iterator iend,
        dlib::resizable_tensor& data
    ) const
    {
        const long nr = ibegin->first.nr();
        const long nc = ibegin->first.nc();
        data.set_size(std::distance(ibegin, iend)*2, 1, nr, nc);

        long offset = nr*nc;
        float* data_ptr = data.host();
        for (auto i = ibegin; i != iend; ++i) {
            for (long r = 0; r < nr; ++r) {
                for (long c = 0; c < nc; ++c) {
                    float* p = data_ptr++;
                    *p = static_cast<float>(i->first(r,c))/256.0;
                    *(p+offset) = static_cast<float>(i->second(r,c))/256.0;
                }
            }
            data_ptr += offset;
        }
    }
};

void serialize(const input_image_pair& item, std::ostream& out)
{
    dlib::serialize("input_image_pair", out);
}

void deserialize(input_image_pair& item, std::istream& in)
{
    std::string version;
    dlib::deserialize(version, in);
    if (version != "input_image_pair") {
        throw dlib::serialization_error("Unexpected version found while deserializing input_image_pair.");
    }
}

std::ostream& operator<<(std::ostream& out, const input_image_pair& item)
{
    out << "input_image_pair";
    return out;
}

void to_xml(const input_image_pair& item, std::ostream& out)
{
    out << "<input_image_pair/>";
}

// ---------------------------------------------------------------------------

class loss_contrastive_ {
public:
    // Here, defining the sample_expansion_factor as 2 makes any network
    // utilizing this loss layer to consider pairs of inputs per label.
    const static unsigned int sample_expansion_factor = 2;
    typedef unsigned char label_type;

    loss_contrastive_(double margin_=1.0, double thresh_=1.0)
        : margin(margin_), thresh(thresh_)
    { }

    double get_label_threshold() const;
    void set_label_threshold(double thresh);

    template <
        typename SUB_TYPE,
        typename label_iterator
        >
    void to_label(
        const dlib::tensor& input_tensor,
        const SUB_TYPE& sub,
        label_iterator iter
    ) const
    {
        const dlib::tensor& output_tensor = sub.get_output();
        DLIB_CASSERT(output_tensor.nr() == 1 && 
                     output_tensor.nc() == 1 ,"");
        DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples(),"");

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

    template <
        typename const_label_iterator,
        typename SUBNET
        >
    double compute_loss_value_and_gradient(
        const dlib::tensor& input_tensor,
        const_label_iterator truth,
        SUBNET& sub
    ) const
    {
        const dlib::tensor& output_tensor = sub.get_output();
        dlib::tensor& grad = sub.get_gradient_input();

        // Enforce contracts that define the loss layer interface. This is not
        // necessary for generating this function, but it is good practice to
        // check that contracts are obeyed by the code.
        DLIB_CASSERT(input_tensor.num_samples() != 0,"");
        DLIB_CASSERT(input_tensor.num_samples()%sample_expansion_factor == 0,"");
        DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples(),"");
        DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples(),"");
        DLIB_CASSERT(output_tensor.nr() == 1 && 
                     output_tensor.nc() == 1,"");
        DLIB_CASSERT(grad.nr() == 1 && 
                     grad.nc() == 1,"");

        const double scale = 2.0/output_tensor.num_samples();
        double loss = 0;
        const float* out_data = output_tensor.host();
        float* g = grad.host();
        for (long i = 0; i < output_tensor.num_samples()/2; i += 2) {
            const float y = *truth++;
            DLIB_CASSERT(y == +1 || y == 0, "y: " << y);

            dlib::matrix<float,0,1> x1, x2;
            x1.set_size(output_tensor.k());
            x2.set_size(output_tensor.k());
            for (long k = 0; k < output_tensor.k(); ++k) {
                x1 = out_data[i*output_tensor.k()+k];
                x2 = out_data[(i+1)*output_tensor.k()+k];
            }

            float d = dlib::length(x1-x2);
            if (y) {
                loss += d*d;
                for (long k = 0; k < output_tensor.k(); ++k) {
                    g[i*output_tensor.k()+k] = scale*(x1(k)-x2(k));
                    g[(i+1)*output_tensor.k()+k] = scale*(x2(k)-x1(k));
                }
            }
            else {
                float temp = margin-d;
                if (temp > 0) {
                    loss += temp*temp;
                    float gscale = -scale*temp/(d+1e-4);
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

        return loss*scale;
    }

    friend std::ostream& operator<<(std::ostream& out, const loss_contrastive_& item)
    {
        out << "loss_contrastive \t ("
            << "margin=" << item.margin
            << ", label_threshold=" << item.thresh
            << ")";
        return out;
    }

    friend void to_xml(const loss_contrastive_& item, std::ostream& out)
    {
        out << "<loss_contrastive"
            << " margin='" << item.margin << "'"
            << " label_threshold='" << item.thresh << "'/>";
    }

    friend void serialize(const loss_contrastive_& item, std::ostream& out)
    {
        dlib::serialize("loss_contrastive", out);
        dlib::serialize(item.margin, out);
        dlib::serialize(item.thresh, out);
    }

    friend void deserialize(loss_contrastive_& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version == "loss_contrastive") {
            dlib::deserialize(item.margin, in);
            dlib::deserialize(item.thresh, in);
        }
    }
private:
    double margin;
    double thresh;
};

template <typename SUBNET>
using loss_contrastive = dlib::add_loss_layer<loss_contrastive_,SUBNET>;

// ---------------------------------------------------------------------------

void create_mnist_siamese_dataset(
    char* mnist_dir,
    std::vector<image_pair>& training_pairs,
    std::vector<unsigned char>& training_labels,
    std::vector<image_pair>& testing_pairs,
    std::vector<unsigned char>& testing_labels
)
{
    std::vector<dlib::matrix<unsigned char> > training_images_;
    std::vector<unsigned long> training_labels_;
    std::vector<dlib::matrix<unsigned char> > testing_images_;
    std::vector<unsigned long> testing_labels_;
    dlib::load_mnist_dataset(mnist_dir, training_images_, training_labels_, testing_images_,  testing_labels_);

    dlib::rand rnd;
    training_pairs.reserve(training_images_.size());
    training_labels.reserve(training_images_.size());
    for (unsigned long i = 0; i < training_images_.size(); ++i) {
        unsigned long idx1 = rnd.get_random_64bit_number() % training_images_.size();
        unsigned long idx2 = rnd.get_random_64bit_number() % training_images_.size();
        while (idx1 == idx2) {
            idx1 = rnd.get_random_64bit_number() % training_images_.size();
            idx2 = rnd.get_random_64bit_number() % training_images_.size();
        }

        training_pairs.push_back(std::make_pair(training_images_[idx1], training_images_[idx2]));
        if (training_labels_[idx1] == training_labels_[idx2]) {
            training_labels.push_back(1);
        }
        else {
            training_labels.push_back(0);
        }
    }

    testing_pairs.reserve( testing_images_.size());
    testing_labels.reserve(testing_images_.size());
    for (unsigned long i = 0; i < testing_images_.size(); ++i) {
        unsigned long idx1 = rnd.get_random_64bit_number() % testing_images_.size();
        unsigned long idx2 = rnd.get_random_64bit_number() % testing_images_.size();
        while (idx1 == idx2) {
            idx1 = rnd.get_random_64bit_number() % testing_images_.size();
            idx2 = rnd.get_random_64bit_number() % testing_images_.size();
        }

        testing_pairs.push_back(std::make_pair(testing_images_[idx1], testing_images_[idx2]));
        if (testing_labels_[idx1] == testing_labels_[idx2]) {
            testing_labels.push_back(1);
        }
        else {
            testing_labels.push_back(0);
        }
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

    // These next statements load the dataset into memory.
    std::vector<image_pair> training_pairs;
    std::vector<unsigned char> training_labels;
    std::vector<image_pair> testing_pairs;
    std::vector<unsigned char> testing_labels;
    create_mnist_siamese_dataset(argv[1], training_pairs, training_labels, testing_pairs, testing_labels);

    using net_type = loss_contrastive<
                         dlib::fc<2,
                         dlib::fc<10,
                         dlib::relu<dlib::fc<500,
                         dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<50,5,5,1,1,
                         dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<20,5,5,1,1,
                         input_image_pair> > > > > > > > > > >;

    net_type net;
    dlib::layer<1>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::fc<2,...
    dlib::layer<2>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::fc<10,...
    dlib::layer<4>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::fc<500,...
    dlib::layer<7>(net).layer_details().set_bias_learning_rate_multiplier(2);  // dlib::con<50,...
    dlib::layer<10>(net).layer_details().set_bias_learning_rate_multiplier(2); // dlib::con<20,...

    std::cout << "This network has " << net.num_layers << " layers in it." << std::endl;
    std::cout << net << std::endl;

    dlib::dnn_trainer<net_type,dlib::sgd> trainer(net, dlib::sgd(0.0,0.9));
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(64);
    trainer.be_verbose();

    trainer.set_synchronization_file("mnist_siamese_sync", std::chrono::seconds(10));

    trainer.train(training_pairs, training_labels);
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
