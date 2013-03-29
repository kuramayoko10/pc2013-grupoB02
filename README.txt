*************** TRABALHO 1 - Calculo do PI ***************

* NOME				NUSP
  Cassiano Kleinert Casagrande	
  Guilherme Simao Gibertoni	7152802
  Rodrigo Vicente Beber		


1) Dependências:
-----------------
* Os códigos implementados neste trabalho utilizam duas dependências
  - Biblioteca <math.h>
  - Biblioteca GNU Multiple Precision Arithmetic Library (GMP)

* A primeira dependência já vem instalada por padrão nos sistemas Unix e Windows.
* Já a segunda precisa ser instalada no sistema alvo
  - Opção 1: Faça o download do código-fonte da biblioteca e seu Makefile a partir do endereço URL: http://gmplib.org
  - Opção 2: Em sistemas Unix com gerenciador de pacotes, procurar pela entrada: libgmp10, libgmp-dev e/ou gmp5


2) Compilação:
-----------------
* Após a instalação da biblioteca GMP, o sistema conterá os arquivos .h e .lib já nas pastas padrão. Por exemplo em sistemas Unix:
  - /usr/lib/libgmp.lib
  - /usr/include/gmp.h

* Para execução dos algoritmos basta compilá-los utilizando o compilador GCC
  seguindo o padrão
	gcc nomeAlgoritmo_tipoImplementação.c -o nome_tipo -lgmp -lm
  onde
	nomeAlgoritmo: gauss / borwein / montecarlo
	tipoAlgoritmo: standard / concurrent

* Alternativamente, usuarios linux poden executar o comando 'make' e todos os códigos serão compilados automaticamente


3) Execução:
-----------------
* Agora que os algoritmos já estão compilados basta executá-los seguindo o padrão
	Unix: 	 ./nome_tipo > nome_tipo_saida.txt
	Windows: nome_tipo > nome_tipo_saida.txt

* Para verificar quantas casas corretas do PI foram obtidas, basta compilar e executar o programa compare.c
	gcc compare.c - o compare
	./compare nome_tipo_saida.txt pi

* O arquivo 'pi' apresenta as 10 milhões de casas corretas do PI de acordo com o armazenado em http://archive.org na internet.


4) Notas Finais:
-----------------
* Na versão atual dos algoritmos apenas a ultima iteração está sendo impressa, 
  pois facilita na hora de executar o programa 'compare' e verificar a corretude
  da saída.

* Para obter a saída das outras iterações, basta descomentar no código-fonte a linha
  //mpf_out_str(…)

