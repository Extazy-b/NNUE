namespace nnue {
class Trainer {
public:
  Trainer(NNUE &model);
  void train_on_batch(const std::vector<std::pair<input_t, label_t>> &);
  void save_weights(const std::string &path);
};
} // namespace nnue
