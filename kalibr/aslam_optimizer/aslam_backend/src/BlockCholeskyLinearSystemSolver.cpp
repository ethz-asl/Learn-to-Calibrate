#include <sparse_block_matrix/linear_solver_cholmod.h>
#include <sparse_block_matrix/linear_solver_spqr.h>
#include <aslam/backend/BlockCholeskyLinearSystemSolver.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <sm/PropertyTree.hpp>

#include <unistd.h>
#include <iostream>

namespace aslam
{
namespace backend
{
BlockCholeskyLinearSystemSolver::BlockCholeskyLinearSystemSolver(const std::string& solver,
                                                                 const BlockCholeskyLinearSolverOptions& options)
  : _options(options), _solverType(solver)
{
  initSolver();
}

BlockCholeskyLinearSystemSolver::BlockCholeskyLinearSystemSolver(const sm::PropertyTree& config)
{
  _solverType = config.getString("solverType", "cholesky");
  // NO OPTIONS CURRENTLY IMPLEMENTED
  // USING C++11 would allow to do constructor delegation and more elegant code
  if (_solverType == "cholesky")
  {
    _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
  }
  else if (_solverType == "spqr")
  {
    _solver.reset(new sparse_block_matrix::LinearSolverQr<Eigen::MatrixXd>());
  }
  else
  {
    std::cout << "Unknown block solver type " << _solverType
              << ". Try \"cholesky\" or \"spqr\"\nDefaulting to cholesky.\n";
    _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
  }
}

BlockCholeskyLinearSystemSolver::~BlockCholeskyLinearSystemSolver()
{
}

void BlockCholeskyLinearSystemSolver::initMatrixStructureImplementation(const std::vector<DesignVariable*>& dvs,
                                                                        const std::vector<ErrorTerm*>& errors,
                                                                        bool useDiagonalConditioner)
{
  if (_solverType == "cholesky")
  {
    _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
  }
  else if (_solverType == "spqr")
  {
    _solver.reset(new sparse_block_matrix::LinearSolverQr<Eigen::MatrixXd>());
  }
  else
  {
    std::cout << "Unknown block solver type " << _solverType
              << ". Try \"cholesky\" or \"spqr\"\nDefaulting to cholesky.\n";
    _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
  }
  _solver->init();
  _useDiagonalConditioner = useDiagonalConditioner;
  _errorTerms = errors;
  std::vector<int> blocks;
  for (size_t i = 0; i < dvs.size(); ++i)
  {
    dvs[i]->setBlockIndex(i);
    blocks.push_back(dvs[i]->minimalDimensions());
  }
  std::partial_sum(blocks.begin(), blocks.end(), blocks.begin());
  // Now we can initialized the sparse Hessian matrix.
  _H._M = SparseBlockMatrix(blocks, blocks);
}

void BlockCholeskyLinearSystemSolver::buildSystem(size_t /* nThreads */, bool useMEstimator)
{
  // \todo make multithreaded. This is complicated as it requires synchronized access to the block matrix.
  //       A little bit of effort should make this possible by initializing the structure and adding
  //       a mutex for each block and having writers for each jacobian that have a list of mutexes.
  //       Save it for later.
  _H._M.clear(false);
  _rhs.setZero();
  std::vector<ErrorTerm*>::iterator it, it_end;
  it = _errorTerms.begin();
  it_end = _errorTerms.end();
  for (; it != it_end; ++it)
  {
    (*it)->buildHessian(_H._M, _rhs, useMEstimator);
  }
}

bool BlockCholeskyLinearSystemSolver::solveSystem(Eigen::VectorXd& outDx)
{
  if (_useDiagonalConditioner)
  {
    Eigen::VectorXd d = _diagonalConditioner.cwiseProduct(_diagonalConditioner);
    // Augment the diagonal
    int rowBase = 0;
    for (int i = 0; i < _H._M.bRows(); ++i)
    {
      Eigen::MatrixXd& block = *_H._M.block(i, i, true);
      SM_ASSERT_EQ_DBG(Exception, block.rows(), block.cols(), "Diagonal blocks are square...right?");
      block.diagonal() += d.segment(rowBase, block.rows());
      rowBase += block.rows();
    }
  }
  // Solve the system
  outDx.resize(_H._M.rows());
  bool solutionSuccess = _solver->solve(_H._M, &outDx[0], &_rhs[0]);
  if (_useDiagonalConditioner)
  {
    // Un-augment the diagonal
    int rowBase = 0;
    for (int i = 0; i < _H._M.bRows(); ++i)
    {
      Eigen::MatrixXd& block = *_H._M.block(i, i, true);
      block.diagonal() -= _diagonalConditioner.segment(rowBase, block.rows());
      rowBase += block.rows();
    }
  }
  if (!solutionSuccess)
  {
    // std::cout << "Solution failed...creating a new solver\n";
    // This seems to help when the CHOLMOD stuff gets into a bad state
    initSolver();
  }

  // In the linear system solver, kalibr will build a problem like H * x = b, where the H is the Hessian block,
  // and the last 8 parameters in x are the quaternion (N=4), translation(N=3) and time offset(N=1).
  // We could add log to print the last 8 parameters and the calculated IMU-Camera transformation to check whether they
  // are the same value. I checked they are the same. Then we extract the corresponding Hessian block by the index.
  // Because Hessian is the sparse matrix, the inverse of the block is the covariance what we needed.

  // modified to print covariance
  if (_H._M.rows() > 100)
  {
    printf("BlockCholeskyLinearSystemSolver::solveSystem()");
    std::cout << "Hessian size = " << _H._M.rows() << " X " << _H._M.cols() << ", block size = " << _H._M.bRows()
              << " X " << _H._M.bCols();
    if (_H._M.rows() > 3000 || _H._M.cols() > 3000)
    {
      std::cout << std::endl;
    }
    else
    {
      std::cout << "|H| = " << _H._M.toDense().norm() << std::endl;
    }
    std::cout << "rhs size = " << _rhs.size() << ", |rhs| = " << _rhs.norm() << std::endl;
    std::cout << "dx size = " << outDx.size() << ", |dx| = " << outDx.norm() << std::endl;
    // last 3-2 block
    int startBlockRow = _H._M.bRows() - 3;
    int startBlockCol = _H._M.bCols() - 3;
    int r0 = _H._M.rowBaseOfBlock(startBlockRow);
    int r1 = _H._M.rowBaseOfBlock(startBlockRow + 2);
    int c0 = _H._M.colBaseOfBlock(startBlockCol);
    int c1 = _H._M.colBaseOfBlock(startBlockCol + 2);
    Eigen::MatrixXd hBlock = _H._M.slice(startBlockRow, startBlockRow + 2, startBlockCol, startBlockCol + 2)->toDense();
    std::cout << "H[" << r0 << ":" << r1 << ", " << c0 << ":" << c1 << "] = " << std::endl << hBlock << std::endl;
    std::cout << "H[" << r0 << ":" << r1 << ", " << c0 << ":" << c1 << "]^-1 = " << std::endl
              << hBlock.inverse() << std::endl;
    // std::cout << "x[" << r0 << ":" << r1 << "] = " << std::endl << xBlock << std::endl;

    char* buffer;
    if ((buffer = getcwd(NULL, 0)) == NULL)
    {
      perror("getcwd error");
    }
    else
    {
      printf("H.txt and H_inverse.txt are both saved at: %s\n", buffer);
      free(buffer);
    }

    // H
    std::ofstream mcfile;
    mcfile.open("./H.txt");
    mcfile << hBlock;
    mcfile.close();
    // H inverse
    std::ofstream mcfile1;
    mcfile1.open("./H_inverse.txt");
    mcfile1 << hBlock.inverse();
    mcfile1.close();
    // x
    std::ofstream mcfile2;
    mcfile2.open("./x.txt");
    mcfile2 << outDx;
    mcfile2.close();
    // b
    std::ofstream mcfile3;
    mcfile3.open("./b.txt");
    mcfile3 << _rhs;
    mcfile3.close();
    // H full
    std::ofstream mcfile4;
    mcfile4.open("./H_full.txt");
    mcfile4 << _H._M;
    mcfile4.close();
  }

  return solutionSuccess;
}

void BlockCholeskyLinearSystemSolver::initSolver()
{
  if (_solverType == "cholesky")
  {
    _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
  }
  else if (_solverType == "spqr")
  {
    _solver.reset(new sparse_block_matrix::LinearSolverQr<Eigen::MatrixXd>());
  }
  else
  {
    std::cout << "Unknown block solver type " << _solverType
              << ". Try \"cholesky\" or \"spqr\"\nDefaulting to cholesky.\n";
    _solver.reset(new sparse_block_matrix::LinearSolverCholmod<Eigen::MatrixXd>());
  }
}

/// \brief compute only the covariance blocks associated with the block indices passed as an argument
void BlockCholeskyLinearSystemSolver::computeCovarianceBlocks(const std::vector<std::pair<int, int> >& blockIndices,
                                                              SparseBlockMatrix& outP)
{
  // Not sure why I have to do this.
  //_solver->init();
  if (_useDiagonalConditioner)
  {
    Eigen::VectorXd d = _diagonalConditioner.cwiseProduct(_diagonalConditioner);
    // Augment the diagonal
    int rowBase = 0;
    for (int i = 0; i < _H._M.bRows(); ++i)
    {
      Eigen::MatrixXd& block = *_H._M.block(i, i, true);
      SM_ASSERT_EQ_DBG(Exception, block.rows(), block.cols(), "Diagonal blocks are square...right?");
      block.diagonal() += d.segment(rowBase, block.rows());
      rowBase += block.rows();
    }
  }
  bool success = _solver->solvePattern(outP, blockIndices, _H._M);
  SM_ASSERT_TRUE(Exception, success, "Unable to retrieve covariance");
  if (_useDiagonalConditioner)
  {
    // Un-augment the diagonal
    int rowBase = 0;
    for (int i = 0; i < _H._M.bRows(); ++i)
    {
      Eigen::MatrixXd& block = *_H._M.block(i, i, true);
      block.diagonal() -= _diagonalConditioner.segment(rowBase, block.rows());
      rowBase += block.rows();
    }
  }
}

void BlockCholeskyLinearSystemSolver::copyHessian(SparseBlockMatrix& H)
{
  _H._M.cloneInto(H);
}

const BlockCholeskyLinearSolverOptions& BlockCholeskyLinearSystemSolver::getOptions() const
{
  return _options;
}

BlockCholeskyLinearSolverOptions& BlockCholeskyLinearSystemSolver::getOptions()
{
  return _options;
}

void BlockCholeskyLinearSystemSolver::setOptions(const BlockCholeskyLinearSolverOptions& options)
{
  _options = options;
}

double BlockCholeskyLinearSystemSolver::rhsJtJrhs()
{
  Eigen::VectorXd JtJrhs;
  _H.rightMultiply(_rhs, JtJrhs);
  return _rhs.dot(JtJrhs);
}

}  // namespace backend
}  // namespace aslam
