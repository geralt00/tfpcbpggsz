int main (int argc, char*argv[]) {
auto bNames = NamedParameter<std::string>("Branches", std::vector<std::string>()
              ,"List of branch names, assumed to be \033[3m daughter1_px ... daughter1_E, daughter2_px ... \033[0m" ).getVector();


    cout << "Hello, World!" << endl;
    cout << "Branches: ";
    for (auto &b : bNames) cout << b << " ";
    cout << endl;

    return 0;
}