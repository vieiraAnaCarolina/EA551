%{
#include <stdlib.h>
#include "ea551-mat-txt-parser.h"
#include "ea551-mat-txt-parser.tab.h"
%}

%option noyywrap yylineno nounput noinput

DIGITO    [0-9]
DIGITOS   {DIGITO}+
SINAL     [+-]
EXPOENTE  [eE]{SINAL}?{DIGITOS}
FRACAO    ({DIGITOS}\.{DIGITO}*)|(\.{DIGITOS})
FLOAT     {SINAL}?(({FRACAO}{EXPOENTE}?)|({DIGITOS}{EXPOENTE}))
INTEIRO   {SINAL}?{DIGITOS}

%%

[ \t\r\n]+  ;

","         { return VIRGULA; }
"["         { return COLCHETE_ESQ; }
"]"         { return COLCHETE_DIR; }

{FLOAT} {
    //Captura a string de um número flutuante e converte para o tipo double
    yylval.dval = strtod(yytext, NULL);
    return FLOAT_LIT;
}

{INTEIRO} {
    //Captura a string, converte para inteiro long e armazena como double
    long v = strtol(yytext, NULL, 10);
    yylval.dval = (double)v;
    if (v < 0 || v > 255) return INT32_LIT;  /* negativo ou >255 → int32 */
    return UINT8_LIT;
}

. { return yytext[0]; }

%%
