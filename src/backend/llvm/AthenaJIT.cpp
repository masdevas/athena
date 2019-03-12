/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/backend/llvm/AthenaJIT.h>

#include <iostream>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

namespace athena::backend::llvm {
AthenaJIT::AthenaJIT(::llvm::orc::JITTargetMachineBuilder JTMB, ::llvm::DataLayout DL) :
    mObjectLayer(mExecutionSession, []() { return ::llvm::make_unique<::llvm::SectionMemoryManager>(); }),
    mCompileLayer(mExecutionSession, mObjectLayer, ::llvm::orc::ConcurrentIRCompiler(std::move(JTMB))),
    mOptimizeLayer(mExecutionSession, mCompileLayer, optimizeModule),
    mDataLayout(std::move(DL)),
    mMangle(mExecutionSession, mDataLayout),
    mContext(::llvm::make_unique<::llvm::LLVMContext>()) {

    mExecutionSession.getMainJITDylib()
        .setGenerator(cantFail(::llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
}
::llvm::Expected<std::unique_ptr<AthenaJIT>> AthenaJIT::create() {
    auto JTMB = ::llvm::orc::JITTargetMachineBuilder::detectHost();

    if (!JTMB)
        return JTMB.takeError();

    auto DL = JTMB->getDefaultDataLayoutForTarget();
    if (!DL)
        return DL.takeError();

    return ::llvm::make_unique<AthenaJIT>(std::move(*JTMB), std::move(*DL));
}
::llvm::Error AthenaJIT::addModule(std::unique_ptr<::llvm::Module> &M) {
//    M->dump();
    return mOptimizeLayer.add(mExecutionSession.getMainJITDylib(),
                              ::llvm::orc::ThreadSafeModule(std::move(M), mContext));
}
::llvm::Expected<::llvm::JITEvaluatedSymbol> AthenaJIT::lookup(::llvm::StringRef Name) {
    std::string o;
    auto stream = ::llvm::raw_string_ostream(o);
    mExecutionSession.dump(stream);
    stream.flush();
    std::cout << o;
    std::cout.flush();
    return mExecutionSession.lookup({&mExecutionSession.getMainJITDylib()}, mMangle(Name.str()));
}
::llvm::Expected<::llvm::orc::ThreadSafeModule>
AthenaJIT::optimizeModule(::llvm::orc::ThreadSafeModule TSM,
                          const ::llvm::orc::MaterializationResponsibility &R) {
    // Create a function pass manager.
    auto FPM = ::llvm::make_unique<::llvm::legacy::FunctionPassManager>(TSM.getModule());

    // Add some optimizations.
    //FPM->add(::llvm::createInstructionCombiningPass());
    FPM->add(::llvm::createReassociatePass());
    FPM->add(::llvm::createGVNPass());
    FPM->add(::llvm::createCFGSimplificationPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : *TSM.getModule())
        FPM->run(F);

    return TSM;
}

}
