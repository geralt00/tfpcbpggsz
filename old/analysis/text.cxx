   TTree *tree_data = tree_data_ori->CloneTree();


   for (int i = 0; i < tree_data->GetEntries(); i++)
    {
        tree_data->GetEntry(i);
        var[0]->setVal(mks_k);
        var[1]->setVal(mks_pi);
        RooArgSet s12_s13 = RooArgSet(*var[0], *var[1]);
        phsppdf_val = PDF_bkg->getVal(s12_s13);
        tree_data->Fill()
    }
    tree_data->Fill()
