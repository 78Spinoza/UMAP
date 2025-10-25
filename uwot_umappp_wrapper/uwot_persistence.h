#pragma once

#include "uwot_model.h"
#include "uwot_hnsw_utils.h"
#include "uwot_progress_utils.h"
#include <iostream>
#include <fstream>

namespace persistence_utils {

    // Main persistence functions
    int save_model(UwotModel* model, const char* filename);
    UwotModel* load_model(const char* filename);

    
}