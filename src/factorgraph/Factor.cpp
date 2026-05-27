#include <iostream>
#include <string>
#include "Factor.h"
#include "Message.h"

namespace factorgraph
{

ThreeMessageFactor::ThreeMessageFactor(
    Message *xin, Message *xout, 
    Message *yin, Message *yout, 
    Message *zin, Message *zout):
    m_x(xin,xout),
    m_y(yin,yout),
    m_z(zin,zout)
{ }

ThreeMessageFactor::ThreeMessageFactor(
    const Message *xin, Message *xout, 
    Message *yin, Message *yout, 
    Message *zin, Message *zout):
    m_x(xin,xout),
    m_y(yin,yout),
    m_z(zin,zout)
{ }

void ThreeMessageFactor::Initialise()
{
    if (m_x.MutableIn()) this->_Initialise(m_x.MutableIn());
    if (m_y.MutableIn()) this->_Initialise(m_y.MutableIn());
    if (m_z.MutableIn()) this->_Initialise(m_z.MutableIn());
}

void ThreeMessageFactor::Update(int which, const char *debug)
{
    if (m_z.Out() && (which==0 || which==3))
    {
        this->_Apply(m_z.Out(), m_x.In(), 1.0, m_y.In(), debug);
    }

    if (m_x.Out() && (which==0 || which==1))
    {
        this->_Apply(m_x.Out(), m_z.In(), -1.0, m_y.In(), debug);
    }

    if (m_y.Out() && (which==0 || which==2))
    {
        this->_Apply(m_y.Out(), m_z.In(), -1.0, m_x.In(), debug);
    }
}


SumFactor::SumFactor(Message *xin, Message *xout, Message *yin, Message *yout, Message *zin, Message *zout, bool initialise):
ThreeMessageFactor(xin,xout,yin,yout,zin,zout)
{
    if (initialise)
    {
        ThreeMessageFactor::Initialise();
    }
}

SumFactor::SumFactor(const Message *xin, Message *xout, Message *yin, Message *yout, Message *zin, Message *zout, bool initialise):
ThreeMessageFactor(xin,xout,yin,yout,zin,zout)
{ 
    if (initialise)
    {
        ThreeMessageFactor::Initialise();
    }
}

void SumFactor::_Initialise(Message *m) const
{
    m->ToConvolveIdentity();
}

void SumFactor::_Apply(Message *m1, const Message *m2, double k, const Message *m3, const char *debug) const
{
    m1->Copy(*m2);
    m1->Convolve(*m3,k);
    if (debug) 
        std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
        m1->AsString()<<"(t) == "<<m2->AsString()<<"(t) * "<<m3->AsString()<<'('<<k<<"t)"<< std::endl;
}

// --------- SumNFactor --------

SumNFactor::SumNFactor(Message *yin, Message *yout, bool initialise):
  m_y(yin,yout)
{
    if (initialise) yin->ToConvolveIdentity();
}

void SumNFactor::PushMessagePair(Message *xin, Message *xout, bool initialise)
{
    if (initialise) xin->ToConvolveIdentity();
    m_x.push_back(MessagePair(xin,xout));
}

void SumNFactor::PushMessagePair(const Message *xin, Message *xout)
{
    m_x.push_back(MessagePair(xin,xout));
}

void SumNFactor::Initialise()
{
    if (m_y.MutableIn()) m_y.MutableIn()->ToConvolveIdentity();
    for (std::size_t i=0; i<m_x.size(); ++i)
    {
        if (m_x[i].MutableIn()) m_x[i].MutableIn()->ToConvolveIdentity();
    }
}

void SumNFactor::Update(int which, const char *debug)
{
    if (m_y.Out() && (which==0 || which==1))
    {
        m_y.Out()->ToConvolveIdentity();
        for (std::size_t i=0; i<m_x.size(); ++i)
        {
            m_y.Out()->Convolve(*m_x[i].In(), 1.0);
        }
        if (debug) 
            std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
            m_y.Out()->AsString()<<std::endl;
    }
    for (std::size_t j=0; j<m_x.size(); ++j)
    {
        MessagePair &xj = m_x[j];
        if (xj.Out() && (which==0 || which==j+2))
        {
            xj.Out()->Copy(*m_y.In());
            for (std::size_t i=0; i<m_x.size(); ++i)
            {
                if (i!=j)
                {
                    xj.Out()->Convolve(*m_x[i].In(),-1.0);
                }
            }
            if (debug) 
                std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
                xj.Out()->AsString()<<std::endl;
        }
    }
}


// --------- EqualityFactor ---------

EqualityFactor::EqualityFactor(Message *xin, Message *xout, Message *yin, Message *yout, Message *zin, Message *zout, bool initialise):
ThreeMessageFactor(xin,xout,yin,yout,zin,zout)
{
    if (initialise)
    {
        ThreeMessageFactor::Initialise();
    }
}

void EqualityFactor::_Initialise(Message *m) const
{
    m->ToMultiplyIdentity();
}

void EqualityFactor::_Apply(Message *m1, const Message *m2, double, const Message *m3, const char *debug) const
{
    m1->Copy(*m2);
    m1->Multiply(*m3,1.0);
    if (debug) 
        std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
        m1->AsString()<<" == "<<m2->AsString()<<" x "<<m3->AsString()<<std::endl;
}


TerminalConstraintFactor::TerminalConstraintFactor(double lb, double ub, Message *min, Message *mout, bool initialise):
m_lb(lb), m_ub(ub), m_message(min,mout)
{
    if (initialise)
    {
        this->Initialise();
    }
}

void TerminalConstraintFactor::Initialise()
{
    if (m_message.MutableIn()) m_message.MutableIn()->ToMultiplyIdentity();
}

void TerminalConstraintFactor::Update(int, const char *debug)
{
    if (m_message.Out())
    {
        Message &mout = *m_message.Out();
        const Message &msgin = *m_message.In();
        mout.Copy(msgin);
        mout.Constrain(m_lb,m_ub);

        if (debug) 
            std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
            msgin.AsString()<<" x TN(0,INF,"<<m_lb<<','<<m_ub<<") ~ "<<mout.AsString()<<" == ";

        mout.Divide(msgin, 1.0);

        if (debug) std::cout<<msgin.AsString()<<" x "<<mout.AsString()<<" == msgIn x msgOut"<<std::endl;
    }
}

InteriorConstraintFactor::InteriorConstraintFactor(
    double lb, double ub, 
    Message *xin, Message *xout,
    Message *yin, Message *yout,
    bool constrainXin2Yout,
    bool constrainYin2Xout,
    bool initialise):
m_lb(lb), m_ub(ub),
m_x(xin,xout),
m_y(yin,yout),
m_constrainXin2Yout(constrainXin2Yout),
m_constrainYin2Xout(constrainYin2Xout)

{
    if (initialise)
    {
        this->Initialise();
    }
}

void InteriorConstraintFactor::Initialise()
{
    if (m_x.MutableIn()) m_x.MutableIn()->ToMultiplyIdentity();
    if (m_y.MutableIn()) m_y.MutableIn()->ToMultiplyIdentity();
}

void InteriorConstraintFactor::Update(int which, const char *debug)
{
    static const bool perVariable = true; // I don't know which is better and/or more principled

    if (m_x.Out() && (which==0 || which==1))
    {
        this->Update0(m_y, m_x, m_constrainYin2Xout, perVariable, debug);
    }
    if (m_y.Out() && (which==0 || which==2))
    {
        this->Update0(m_x, m_y, m_constrainXin2Yout, perVariable, debug);
    }
}

void InteriorConstraintFactor::Update0(const MessagePair &u, MessagePair &v, bool doConstrain, bool perVariable, const char *debug)
{
    const Message &uIn = *u.In();
    const Message &vIn = *v.In();
    Message &vOut = *v.Out();

    if (perVariable && doConstrain)
    {
        vOut.Copy(vIn);
        if (!uIn.IsMultiplyIdentity())
        {
            //vOut.ConstrainSolve(m_lb,m_ub);
            vOut.Multiply(uIn,1.0);
            vOut.Constrain(m_lb,m_ub);
        }

        if (debug) 
            std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
            uIn.AsString()<<" x " << vIn.AsString() << " x TN(0,INF,"<<m_lb<<','<<m_ub<<") ~ "<<vOut.AsString()<<" == ";

        vOut.Divide(vIn, 1.0);

        if (debug) 
            std::cout<<vIn.AsString()<<" x "<<vOut.AsString()<<" == msgIn x msgOut"<<std::endl;
    }
    else if (!perVariable && doConstrain)
    {
        vOut.Copy(uIn);
        vOut.Constrain(m_lb,m_ub);

        if (debug) 
            std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
            vOut.AsString() << " == " << uIn.AsString()<<" x TN(0,INF,"<<m_lb<<','<<m_ub<<")";
    }
    else
    {
        vOut.Copy(uIn);
        if (debug)
            std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
            uIn.AsString()<<" == " << vIn.AsString();
    }
}

TransformFactor::TransformFactor(
    const Eigen::MatrixXd &A,
    Message *xin, Message *xout,
    Message *yin, Message *yout,
    bool initialise):
m_A(A),
m_svdA(A, Eigen::ComputeFullU | Eigen::ComputeFullV),
m_x(xin,xout),
m_y(yin,yout)
{
    if (initialise)
    {
        this->Initialise();
    }
}

void TransformFactor::Initialise()
{
    if (m_x.MutableIn()) m_x.MutableIn()->ToConvolveIdentity();
    if (m_y.MutableIn()) m_y.MutableIn()->ToMultiplyIdentity();
}

void TransformFactor::Update(int which, const char *debug)
{
    if (m_y.Out() && (which==0 || which==2))
    {
        m_y.Out()->Copy(*m_x.In());
        m_y.Out()->ForwardTransform(m_A);
        if (debug)
        {
            Eigen::IOFormat fmt(3, Eigen::DontAlignCols, " ", ";", "", "", "[", "]");
            std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
                m_y.Out()->AsString()<<" == "<<m_A.format(fmt)<<" x "<<m_x.In()->AsString()<<std::endl;
        }

    }
    if (m_x.Out() && (which==0 || which==1))
    {
        m_x.Out()->Copy(*m_y.In());
        m_x.Out()->ReverseTransform(m_A, &m_svdA);
        if (debug)
        {
            Eigen::IOFormat fmt(3, Eigen::DontAlignCols, " ", ";", "", "", "[", "]");
            std::cout<<"Factor("<<this->Type()<<",'"<<debug<<"'): "<<
                m_x.Out()->AsString()<<" == "<<m_A.format(fmt)<<" \\ "<<m_y.In()->AsString()<<std::endl;
        }
    }
}

}
