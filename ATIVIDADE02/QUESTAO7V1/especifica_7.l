%{
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
/* Inclui o cabeçalho gerado pelo Bison contendo os tokens (CHAVE, VALOR) */
#include "especifica_7.tab.h" 
%}

/* Desativa a necessidade da biblioteca libfl/yywrap ao atingir o fim do arquivo (EOF) */
%option noyywrap

%%

/* 1. Ignora caracteres de espaço em branco, tabulação e quebras de linha */
[ \t\r\n]+             { }

/* 2. Ignora comentários do YAML iniciados por '#' até o fim da linha */
"#" [^\n]*            { }

/* 3. Delimitadores estruturais: retornam seu próprio valor ASCII como caractere literal */
":"                   { return ':'; }
"["                   { return '['; }
"]"                   { return ']'; }
","                   { return ','; }

/* 4. Reconhece números inteiros e ponto flutuante com sinal opcional */
/* NOTA DE VAZAMENTO: O strdup aloca memória dinamicamente no Heap para yylval.str */
[-+]?[0-9]+(\.[0-9]+)? { 
    yylval.str = strdup(yytext); 
    return VALOR; 
}

/* 5. Reconhece strings entre aspas duplas retas ou curvas/tipográficas */
/* NOTA DE VAZAMENTO: O strdup aloca memória no Heap sem desalocação posterior */
\"([^\"\n])*\"|“([^”\n])*” { 
    yylval.str = strdup(yytext); 
    return VALOR; 
}

/* 6. Reconhece identificadores (nomes de chaves alfanuméricas) */
/* NOTA DE VAZAMENTO: Aloca memória para o nome da chave via strdup */
[a-zA-Z_][a-zA-Z0-9_]* { 
    yylval.str = strdup(yytext); 
    return CHAVE; 
}

/* 7. Regra coringa: ignora qualquer outro caractere não mapeado */
.                     { }

%%