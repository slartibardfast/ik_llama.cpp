// Standalone Vulkan reproducer for ACO/RADV specialization-constant
// f32 nondeterminism on RDNA2. See README.md for background.
//
// No ggml / llama.cpp dependencies. Links against libvulkan only.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <vector>
#include <string>

static void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
    std::exit(2);
}

#define VKCHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) die("VkResult=%d at %s:%d (%s)", _r, __FILE__, __LINE__, #x); } while (0)

static std::vector<uint32_t> read_spirv(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot open SPIR-V file: %s", path.c_str());
    std::streamsize n = f.tellg();
    if (n <= 0 || (n % 4) != 0) die("bad SPIR-V size %lld", (long long)n);
    f.seekg(0);
    std::vector<uint32_t> out(n / 4);
    f.read(reinterpret_cast<char *>(out.data()), n);
    return out;
}

static uint32_t find_compute_queue_family(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, qfs.data());
    for (uint32_t i = 0; i < n; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    }
    die("no compute queue family");
    return 0;
}

static uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t type_bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    die("no mem type for flags 0x%x bits 0x%x", (unsigned)want, type_bits);
    return 0;
}

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void * mapped = nullptr;
};

static Buf make_buf(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size) {
    Buf b{};
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKCHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_mem_type(pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKCHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
    VKCHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    VKCHECK(vkMapMemory(dev, b.mem, 0, size, 0, &b.mapped));
    return b;
}

static void destroy_buf(VkDevice dev, Buf & b) {
    if (b.mapped) vkUnmapMemory(dev, b.mem);
    if (b.buf)   vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem)   vkFreeMemory(dev, b.mem, nullptr);
    b = {};
}

// Same deterministic RNG for weight/activation fill across both runs.
static uint32_t lcg(uint32_t & s) { s = s * 1664525u + 1013904223u; return s; }
static float rand_float(uint32_t & s) {
    // Uniform in [-1, 1]
    uint32_t x = lcg(s);
    return (float(int32_t(x)) / float(int32_t(0x80000000u))) * 1.0f;
}

struct RunResult {
    std::vector<float> y; // size NUM_COLS
};

static RunResult run_pipeline(
    VkDevice dev, VkPhysicalDevice pd, VkQueue q, uint32_t qf,
    const std::vector<uint32_t> & spv, uint32_t num_cols)
{
    // Must match shader.comp
    const uint32_t K_ELEMS = 4096;
    const uint32_t BLOCK   = 32;
    const uint32_t NBLOCKS = K_ELEMS / BLOCK;
    const uint32_t MAX_COLS = 4;
    const uint32_t NUM_ROWS = 2;

    // Layout:
    //   w_packed : int32[MAX_COLS * NUM_ROWS * K_ELEMS/4]
    //   scales   : f16  [MAX_COLS * NUM_ROWS * NBLOCKS]
    //   b_qs     : int32[MAX_COLS * K_ELEMS/4]
    //   b_ds     : f16  [MAX_COLS * NBLOCKS]
    //   y        : f32  [MAX_COLS * NUM_ROWS]
    const VkDeviceSize w_bytes   = VkDeviceSize(MAX_COLS) * NUM_ROWS * (K_ELEMS / 4) * sizeof(int32_t);
    const VkDeviceSize sc_bytes  = VkDeviceSize(MAX_COLS) * NUM_ROWS * NBLOCKS * sizeof(uint16_t);
    const VkDeviceSize bqs_bytes = VkDeviceSize(MAX_COLS) * (K_ELEMS / 4) * sizeof(int32_t);
    const VkDeviceSize bds_bytes = VkDeviceSize(MAX_COLS) * NBLOCKS * sizeof(uint16_t);
    const VkDeviceSize y_bytes   = VkDeviceSize(MAX_COLS) * NUM_ROWS * sizeof(float);

    Buf w   = make_buf(dev, pd, w_bytes);
    Buf sc  = make_buf(dev, pd, sc_bytes);
    Buf bqs = make_buf(dev, pd, bqs_bytes);
    Buf bds = make_buf(dev, pd, bds_bytes);
    Buf y   = make_buf(dev, pd, y_bytes);

    // f32 -> f16 round-to-nearest-even helper (only subnormals collapse to 0)
    auto f32_to_f16 = [](float f) -> uint16_t {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        uint32_t sign = (u >> 31) & 0x1u;
        int32_t  exp  = int32_t((u >> 23) & 0xffu) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x3ffu;
        if (exp <= 0)       return uint16_t(sign << 15);
        if (exp >= 31)      return uint16_t((sign << 15) | (31u << 10));
        return uint16_t((sign << 15) | (uint32_t(exp) << 10) | mant);
    };

    // Fill deterministically. Column 0 (j=0) is generated first so its
    // byte pattern is identical regardless of which pipeline reads it.
    {
        uint32_t seed = 0xC0FFEEu;
        int8_t * wp = static_cast<int8_t *>(w.mapped);
        for (uint32_t j = 0; j < MAX_COLS; ++j) {
            for (uint32_t r = 0; r < NUM_ROWS; ++r) {
                for (uint32_t k = 0; k < K_ELEMS; ++k) {
                    uint32_t rr = lcg(seed);
                    int v = int(rr >> 24) - 128;
                    if (v < -127) v = -127;
                    wp[(j * NUM_ROWS + r) * K_ELEMS + k] = int8_t(v);
                }
            }
        }
        uint32_t seed_s = 0xBADF00Du;
        uint16_t * sp = static_cast<uint16_t *>(sc.mapped);
        for (uint32_t j = 0; j < MAX_COLS; ++j) {
            for (uint32_t r = 0; r < NUM_ROWS; ++r) {
                for (uint32_t b = 0; b < NBLOCKS; ++b) {
                    float f = 0.01f + 0.99f * (float(lcg(seed_s) >> 8) / float(1 << 24));
                    sp[(j * NUM_ROWS + r) * NBLOCKS + b] = f32_to_f16(f);
                }
            }
        }
        uint32_t seed_b = 0xDEADBEEFu;
        int8_t * bqp = static_cast<int8_t *>(bqs.mapped);
        for (uint32_t j = 0; j < MAX_COLS; ++j) {
            for (uint32_t k = 0; k < K_ELEMS; ++k) {
                uint32_t rr = lcg(seed_b);
                int v = int(rr >> 24) - 128;
                if (v < -127) v = -127;
                bqp[j * K_ELEMS + k] = int8_t(v);
            }
        }
        uint32_t seed_bs = 0xFEEDFACEu;
        uint16_t * bdsp = static_cast<uint16_t *>(bds.mapped);
        for (uint32_t j = 0; j < MAX_COLS; ++j) {
            for (uint32_t b = 0; b < NBLOCKS; ++b) {
                float f = 0.01f + 0.99f * (float(lcg(seed_bs) >> 8) / float(1 << 24));
                bdsp[j * NBLOCKS + b] = f32_to_f16(f);
            }
        }
        std::memset(y.mapped, 0, y_bytes);
    }

    // Descriptor set layout / pool / set (5 bindings: w, scales, b_qs, b_ds, y)
    VkDescriptorSetLayoutBinding bindings[5] = {};
    for (int i = 0; i < 5; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 5;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

    VkDescriptorBufferInfo dbis[5] = {
        {w.buf,   0, VK_WHOLE_SIZE},
        {sc.buf,  0, VK_WHOLE_SIZE},
        {bqs.buf, 0, VK_WHOLE_SIZE},
        {bds.buf, 0, VK_WHOLE_SIZE},
        {y.buf,   0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[5] = {};
    for (int i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(dev, 5, writes, 0, nullptr);

    // Pipeline layout (with 4-byte push constant: rt_num_rows)
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout plo;
    VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &plo));

    // Shader module
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * sizeof(uint32_t);
    smci.pCode = spv.data();
    VkShaderModule sm;
    VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    // Specialization: NUM_COLS (constant_id=0)
    VkSpecializationMapEntry sme{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo si{};
    si.mapEntryCount = 1;
    si.pMapEntries = &sme;
    si.dataSize = sizeof(uint32_t);
    si.pData = &num_cols;

    // Require wave32 on RDNA2 so shader NSUB=4 (LSIZE=128 / 32) matches.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rssi{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rssi.requiredSubgroupSize = 32;

    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.pNext = &rssi;
    ssci.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = sm;
    ssci.pName = "main";
    ssci.pSpecializationInfo = &si;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci;
    cpci.layout = plo;
    VkPipeline pipe;
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    // Command pool / buffer
    VkCommandPoolCreateInfo cpinfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpinfo.queueFamilyIndex = qf;
    VkCommandPool cpool;
    VKCHECK(vkCreateCommandPool(dev, &cpinfo, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VKCHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1, &dset, 0, nullptr);
    // Push runtime num_rows = NUM_ROWS (= 2). Runtime value forces ACO
    // to keep the row loop live rather than collapsing it.
    const uint32_t rt_rows = NUM_ROWS;
    vkCmdPushConstants(cb, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &rt_rows);
    vkCmdDispatch(cb, 1, 1, 1);
    VKCHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo subi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    subi.commandBufferCount = 1;
    subi.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VKCHECK(vkCreateFence(dev, &fci, nullptr, &fence));
    VKCHECK(vkQueueSubmit(q, 1, &subi, fence));
    VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

    RunResult r;
    r.y.resize(num_cols * NUM_ROWS);
    std::memcpy(r.y.data(), y.mapped, num_cols * NUM_ROWS * sizeof(float));

    // Tear down
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, plo, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    destroy_buf(dev, w);
    destroy_buf(dev, sc);
    destroy_buf(dev, bqs);
    destroy_buf(dev, bds);
    destroy_buf(dev, y);
    return r;
}

int main(int argc, char ** argv) {
    const char * spv_path = (argc > 1) ? argv[1] : "shader.spv";

    // Instance
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "mesa-repro";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst));

    // Pick device 0
    uint32_t ndev = 0;
    VKCHECK(vkEnumeratePhysicalDevices(inst, &ndev, nullptr));
    if (ndev == 0) die("no Vulkan devices");
    std::vector<VkPhysicalDevice> pds(ndev);
    VKCHECK(vkEnumeratePhysicalDevices(inst, &ndev, pds.data()));
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device[0]: %s (driver 0x%x, vendor 0x%x)\n",
                props.deviceName, props.driverVersion, props.vendorID);

    uint32_t qf = find_compute_queue_family(pd);
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = qf;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &qprio;
    // Enable subgroup-size-controlled compute (wave32 on RDNA2).
    VkPhysicalDeviceSubgroupSizeControlFeatures ssc{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    ssc.subgroupSizeControl = VK_TRUE;
    ssc.computeFullSubgroups = VK_TRUE;

    // f16 and 8-bit storage for the q8_0-style dequant path.
    VkPhysicalDeviceShaderFloat16Int8Features f16i8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16i8.shaderFloat16 = VK_TRUE;
    f16i8.shaderInt8 = VK_TRUE;
    f16i8.pNext = &ssc;

    VkPhysicalDevice16BitStorageFeatures st16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    st16.storageBuffer16BitAccess = VK_TRUE;
    st16.pNext = &f16i8;

    VkPhysicalDevice8BitStorageFeatures st8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    st8.storageBuffer8BitAccess = VK_TRUE;
    st8.pNext = &st16;

    VkPhysicalDeviceShaderIntegerDotProductFeatures idp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    idp.shaderIntegerDotProduct = VK_TRUE;
    idp.pNext = &st8;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &idp;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    const char * dev_exts[] = {
        "VK_EXT_subgroup_size_control",
        "VK_KHR_shader_float16_int8",
        "VK_KHR_16bit_storage",
        "VK_KHR_8bit_storage",
        "VK_KHR_shader_integer_dot_product",
    };
    dci.enabledExtensionCount = 5;
    dci.ppEnabledExtensionNames = dev_exts;
    VkDevice dev;
    VKCHECK(vkCreateDevice(pd, &dci, nullptr, &dev));

    // Probe subgroup properties.
    VkPhysicalDeviceSubgroupSizeControlProperties sscp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sscp;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    std::printf("subgroup size range: [%u..%u], required stages mask=0x%x\n",
                sscp.minSubgroupSize, sscp.maxSubgroupSize, sscp.requiredSubgroupSizeStages);
    VkQueue q;
    vkGetDeviceQueue(dev, qf, 0, &q);

    auto spv = read_spirv(spv_path);

    auto dump = [](const char * tag, const RunResult & r) {
        std::printf("--- %s\n", tag);
        for (size_t i = 0; i < r.y.size(); ++i) {
            uint32_t u; std::memcpy(&u, &r.y[i], 4);
            std::printf("  y[%zu] = %.9g  (bits=0x%08x)\n", i, r.y[i], u);
        }
    };

    RunResult a1 = run_pipeline(dev, pd, q, qf, spv, 1);
    dump("pipeline A: NUM_COLS=1", a1);
    RunResult a2 = run_pipeline(dev, pd, q, qf, spv, 2);
    dump("pipeline B: NUM_COLS=2", a2);
    RunResult a4 = run_pipeline(dev, pd, q, qf, spv, 4);
    dump("pipeline C: NUM_COLS=4", a4);

    // Compare column-0 (first NUM_ROWS floats) across all three variants.
    const size_t ROWS = a1.y.size(); // = 1 * NUM_ROWS
    bool all_same = true;
    for (size_t i = 0; i < ROWS; ++i) {
        bool eq12 = std::memcmp(&a1.y[i], &a2.y[i], 4) == 0;
        bool eq14 = std::memcmp(&a1.y[i], &a4.y[i], 4) == 0;
        uint32_t u1, u2, u4;
        std::memcpy(&u1, &a1.y[i], 4);
        std::memcpy(&u2, &a2.y[i], 4);
        std::memcpy(&u4, &a4.y[i], 4);
        std::printf("col0 row %zu: 1=0x%08x  2=0x%08x (ulp %+d)  4=0x%08x (ulp %+d)\n",
                    i, u1, u2, int32_t(u2) - int32_t(u1),
                    u4, int32_t(u4) - int32_t(u1));
        if (!eq12 || !eq14) all_same = false;
    }
    std::printf("--- result: column 0 %s across NUM_COLS={1,2,4}\n",
                all_same ? "BYTE-IDENTICAL" : "DIVERGENT");
    bool same = all_same;

    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return same ? 0 : 1;
}
