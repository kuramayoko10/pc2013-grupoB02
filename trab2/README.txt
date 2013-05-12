*************** TRABALHO 2 - Classificacaoo de Palíndromos ***************

* NOME					NUSP
  Cassiano Kleinert Casagrande		7152819
  Guilherme Simao Gibertoni		7152802
  Rodrigo Vicente Beber		7152490

* Endereço do repositório de código: [GIT] https://github.com/kuramayoko10/pc2013-grupoB02.git

1) Dependências:
-----------------
* Os códigos implementados neste trabalho utilizam três dependências
  - Biblioteca <math.h>
  - Biblioteca OpenMP
  - Biblioteca OpenMPI (MPI)

* Essas dependências devem vir instalads por padrão nos sistemas UNIX.
* Caso seja necessário reinstalação:
    1) OpenMP:
    2) OpenMPI: 


2) Compilação:
-----------------
* Após a instalação das bibliotecas necessarias, basta compilar utilizando makefile
	make all


3) Execução:
-----------------
* Agora que os algoritmos já estão compilados basta executá-los seguindo o padrão
	Sequencial: 	./seq arquivo_entrada modo_leitura
	OpenMP: 	./omp arquivo_entrada
	MPI:		./mpi arquivo_entrada

* O arquivo 'pi' apresenta as 10 milhões de casas corretas do PI de acordo com o armazenado em http://archive.org na internet.


4) Notas Finais:
-----------------
* Na versão atual dos algoritmos apenas a ultima iteração está sendo impressa, 
  pois facilita na hora de executar o programa 'compare' e verificar a corretude
  da saída.

* Para obter a saída das outras iterações, basta descomentar no código-fonte a linha
  //mpf_out_str(…)

