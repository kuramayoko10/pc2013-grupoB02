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

* Os algoritmos Sequencial e OpenMP irao apresentar a saida na tela. 
* Ja o MPI vai apresentar saida em arquivos de 1 a 13 com nome i-saida.txt
  * Cada arquivo corresponde ao processamento de um dado no do cluster

4) Notas Finais:
-----------------
* 

