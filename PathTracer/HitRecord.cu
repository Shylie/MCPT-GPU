#include "HitRecord.h"
#include "Material.h"

void HitRecord::SetMaterialHost(Material* value)
{
	_mat = value;
	if (value != nullptr) _mat_d = value->GetPtrGPU();
}